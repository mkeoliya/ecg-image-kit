"""Persistent isolated ECG-Image-Kit worker (upstream tree stays unchanged)."""
import argparse
import copy
import contextlib
import json
import os
from pathlib import Path
import random
import sys


"""Lossless RGB PNG with filter None, for measured low-entropy ECG rasters."""
import io
import struct
import zlib
import numpy as np


def encode_rgb(array, *, level=1, strategy=zlib.Z_HUFFMAN_ONLY):
    array=np.asarray(array)
    if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3:
        raise ValueError('PNG encoder expects uint8 RGB')
    height,width,_=array.shape
    rows=np.empty((height,width*3+1),dtype=np.uint8)
    rows[:,0]=0
    rows[:,1:]=array.reshape(height,width*3)
    compressor=zlib.compressobj(level,zlib.DEFLATED,15,strategy=strategy)
    data=compressor.compress(memoryview(rows))+compressor.flush()
    output=io.BytesIO();output.write(b'\x89PNG\r\n\x1a\n')
    def chunk(tag,data):
        output.write(struct.pack('>I',len(data)));output.write(tag);output.write(data)
        output.write(struct.pack('>I',zlib.crc32(data,zlib.crc32(tag))&0xffffffff))
    chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
    chunk(b'IDAT',data);chunk(b'IEND',b'')
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config')
    parser.add_argument('--server', action='store_true')
    args = parser.parse_args()
    code = args.kit.absolute() / 'codes/ecg-image-generator'
    sys.path.insert(0, str(code))
    os.chdir(code)
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    from scipy.stats import bernoulli
    import wfdb
    from extract_leads import get_paper_ecg
    from helper_functions import read_config_file
    import inspect
    optimized = get_paper_ecg.__globals__['ecg_plot'].__globals__.get('SUPPORTS_INLINE_AUGMENTATION', False)
    cached_settings = read_config_file(str(code / 'config.yaml')) if optimized else None
    cached_renderer = None

    def render(job):
        config = job['config']
        record, output = Path(job['record']), Path(job['output'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        header = wfdb.rdheader(str(record))
        settings = copy.deepcopy(cached_settings) if optimized else read_config_file(str(code / 'config.yaml'))
        settings['png_compress_level'] = config.get('png_compress_level', 6)
        settings['png_compress_type'] = config.get('png_compress_type', -1)
        settings['paper_len'] = min(config['duration_s'], header.sig_len / header.fs)
        settings['abs_lead_step'] = settings['paper_len']
        output.mkdir(parents=True, exist_ok=True)
        augmented = config.get('image_noise_std', 0) or config.get('blur_radius', 0) or config.get('jpeg_quality', 100) < 100
        inline = augmented and get_paper_ecg.__globals__['ecg_plot'].__globals__.get('SUPPORTS_INLINE_AUGMENTATION', False)
        if inline:
            from PIL import Image, ImageFilter
            import io
            rng = np.random.default_rng(config['seed'])
            def save_augmented(rgba, path):
                pixels = Image.fromarray(rgba).convert('RGB')
                if config.get('image_noise_std', 0):
                    array = np.asarray(pixels, dtype=np.float32)
                    for row_start in range(0, len(array), 64):
                        chunk = array[row_start:row_start+64]
                        chunk += rng.normal(0, config['image_noise_std'], chunk.shape)
                    np.clip(array, 0, 255, out=array)
                    pixels = Image.fromarray(array.astype(np.uint8))
                if config.get('blur_radius', 0):
                    pixels = pixels.filter(ImageFilter.GaussianBlur(config['blur_radius']))
                if config.get('jpeg_quality', 100) < 100:
                    compressed = io.BytesIO()
                    pixels.save(compressed, format='JPEG', quality=config['jpeg_quality'])
                    compressed.seek(0)
                    with Image.open(compressed) as decoded:
                        pixels = decoded.convert('RGB')
                if config.get('png_encoder') == 'nofilter':
                    strategy=config.get('png_compress_type', 0)
                    if strategy == -1: strategy=0
                    Path(path).write_bytes(encode_rgb(np.asarray(pixels), level=config.get('png_compress_level', 6), strategy=strategy))
                else:
                    pixels.save(path, compress_level=config.get('png_compress_level', 6),
                                compress_type=config.get('png_compress_type', -1))
            settings['_save_augmented_image'] = save_augmented
            settings['_image_augmentation'] = {key:config.get(key) for key in ('image_noise_std','blur_radius','jpeg_quality','seed')}
        # Non-standard lead counts are forced into four columns by upstream;
        # use one column to avoid truncated/empty traces via a local function
        # override, leaving the pinned source tree unchanged.
        nonlocal cached_renderer
        source = inspect.getsource(get_paper_ecg) if cached_renderer is None else None
        if cached_renderer is None:
            source = source.replace("        columns = 4\n        full_mode = 'None'", "        columns = columns if columns != -1 else 1\n        full_mode = 'None'")
            source = source.replace("if full_mode not in full_leads:", "if full_mode != 'None' and full_mode not in full_leads:")
            namespace = dict(get_paper_ecg.__globals__)
            # Targets already have calibrated snapshots; Kit's additional WFDB
            # exporter crashes without a rhythm strip and is not needed here.
            namespace['write_wfdb_file'] = lambda *args, **kwargs: None
            exec(compile(source, '<kit-layout-adapter>', 'exec'), namespace)
            renderer = namespace['get_paper_ecg']
            if optimized:
                cached_renderer = renderer
        else:
            renderer = cached_renderer
        with contextlib.redirect_stdout(sys.stderr):
            renderer(input_file=str(record.with_suffix('.dat')), header_file=str(record.with_suffix('.hea')),
                     output_directory=str(output), seed=config['seed'],
                     add_dc_pulse=bernoulli(int(config.get('calibration_pulse', True))),
                     add_bw=bernoulli(int(config['black_white'])), show_grid=bernoulli(int(config['grid'])),
                     add_print=bernoulli(0), configs=settings, start_index=0, store_configs=2,
                     store_text_bbox=True, resolution=config['resolution'], pad_inches=0, bbox=True,
                     columns=config['columns'], full_mode=config['rhythm_lead'],
                     font_type=str(code / 'Fonts/Times_New_Roman.ttf'), standard_colours=config['grid_color'])
        import matplotlib.pyplot as plt
        plt.close('all')
        if augmented and not inline:
            from PIL import Image, ImageFilter
            import io
            rng = np.random.default_rng(config['seed'])
            for path in sorted(output.glob('*.png')):
                with Image.open(path) as original:
                    pixels = original.convert('RGB')
                if config.get('image_noise_std', 0):
                    array = np.asarray(pixels, dtype=np.float32)
                    array += rng.normal(0, config['image_noise_std'], array.shape)
                    pixels = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
                if config.get('blur_radius', 0):
                    pixels = pixels.filter(ImageFilter.GaussianBlur(config['blur_radius']))
                if config.get('jpeg_quality', 100) < 100:
                    compressed = io.BytesIO()
                    pixels.save(compressed, format='JPEG', quality=config['jpeg_quality'])
                    compressed.seek(0)
                    with Image.open(compressed) as decoded:
                        pixels = decoded.convert('RGB')
                pixels.save(path, compress_level=config.get('png_compress_level', 6),
                            compress_type=config.get('png_compress_type', -1))
                annotations = json.loads(path.with_suffix('.json').read_text())
                annotations['image_augmentation'] = {key: config.get(key) for key in
                    ('image_noise_std', 'blur_radius', 'jpeg_quality', 'seed')}
                path.with_suffix('.json').write_text(json.dumps(annotations) + '\n')
        return {'ok': True}

    if args.server:
        for line in sys.stdin:
            try:
                result = render(json.loads(line))
            except Exception as error:
                import traceback
                traceback.print_exc(file=sys.stderr)
                result = {'ok': False, 'error': str(error)}
            print(json.dumps(result), flush=True)
    else:
        render({'record': str(args.record), 'output': str(args.output), 'config': json.loads(args.config)})


if __name__ == '__main__':
    main()
