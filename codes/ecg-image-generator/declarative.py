"""Render arbitrary declared ECG panels on a shared acquired signal clock.

Layouts are ordinary pages/panels dictionaries. No preset names or lead counts
are used to choose geometry, polarity, timing, speed or amplitude calibration.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageFilter

from ecg_plot import standard_major_colors, standard_minor_colors

_CACHE = {}


def _dictionary(value):
    return asdict(value) if is_dataclass(value) else value


def page_geometry(layout, page_number, speed_mm_s, *, calibration_pulse=True,
                  default_row_height_mm=30., footer_mm=8.):
    """Solve grid placement from span/duration constraints in physical mm."""
    layout = _dictionary(layout)
    page = _dictionary(layout['pages'][page_number])
    panels = [_dictionary(p) for p in layout['panels'] if _dictionary(p)['page'] == page_number]
    if not panels:
        raise ValueError(f'Page {page_number} has no panels')
    rows, columns = page['rows'], page['columns']
    margin, gap = page.get('margin_mm', 10.), page.get('column_gap_mm', 5.)
    row_height = page.get('row_height_mm') or default_row_height_mm
    pulse_width = .2 * speed_mm_s if calibration_pulse else 0.
    calibration_area = pulse_width + 2. if calibration_pulse else 0.
    # Uniform columns satisfy all panel spans, including arbitrary rhythm rows.
    column_width = max((p['duration_s'] * speed_mm_s + 2.
                        - gap * (p.get('column_span', 1) - 1)) / p.get('column_span', 1)
                       for p in panels)
    column_width = max(column_width, 2.)
    width_mm = 2*margin + calibration_area + columns*column_width + (columns-1)*gap
    height_mm = 2*margin + rows*row_height + footer_mm
    positioned = []
    for p in panels:
        q = dict(p)
        q['x_mm'] = margin + calibration_area + p['column']*(column_width+gap)
        q['baseline_mm'] = margin + (p['row'] + p.get('row_span', 1)/2)*row_height
        q['panel_width_mm'] = p.get('column_span', 1)*column_width + (p.get('column_span', 1)-1)*gap
        q['panel_height_mm'] = p.get('row_span', 1)*row_height
        positioned.append(q)
    return dict(width_mm=width_mm, height_mm=height_mm, column_width_mm=column_width,
                row_height_mm=row_height, margin_mm=margin, column_gap_mm=gap,
                calibration_area_mm=calibration_area, pulse_width_mm=pulse_width,
                panels=positioned, rows=rows, columns=columns)


def _canvas(geometry, resolution, speed, gain, style):
    ppmm = resolution/25.4
    width_px = int(np.ceil(geometry['width_mm']*ppmm))
    height_px = int(np.ceil(geometry['height_mm']*ppmm))
    width_mm, height_mm = width_px/ppmm, height_px/ppmm
    key = (width_px, height_px, resolution, bool(style['grid']), bool(style['black_white']),
           style['grid_color'], float(speed), float(gain), style['font_path'])
    if key in _CACHE:
        fig, ax, canvas, background = _CACHE[key]
        for artist in list(ax.lines) + list(ax.texts):
            artist.remove()
        canvas.restore_region(background)
        return fig, ax, canvas, width_mm, height_mm, background
    fig = Figure(figsize=(np.nextafter(width_px/resolution,np.inf), np.nextafter(height_px/resolution,np.inf)), dpi=resolution, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0,0,1,1))
    ax.set_xlim(0,width_mm);ax.set_ylim(height_mm,0);ax.set_axis_off()
    if style['grid']:
        color = style['grid_color']
        major = (.4,.4,.4) if style['black_white'] else standard_major_colors[f'colour{color}']
        minor = (.75,.75,.75) if style['black_white'] else standard_minor_colors[f'colour{color}']
        for step, color, linewidth in ((1.,minor,.15),(5.,major,.45)):
            segments = [[(x,0),(x,height_mm)] for x in np.arange(0,width_mm+.001,step)]
            segments += [[(0,y),(width_mm,y)] for y in np.arange(0,height_mm+.001,step)]
            ax.add_collection(LineCollection(segments, colors=[color], linewidths=linewidth, zorder=1))
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    if len(_CACHE) >= 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (fig,ax,canvas,background)
    return fig,ax,canvas,width_mm,height_mm,background


def _box(points):
    low, high = np.min(points,axis=0), np.max(points,axis=0)
    y1,x1=low;y2,x2=high
    return {0:[float(y1),float(x1)],1:[float(y1),float(x2)],
            2:[float(y2),float(x2)],3:[float(y2),float(x1)]}


def render_layout(values, fs, lead_names, layout, output_directory, record_name,
                  *, speed_mm_s=25., gain_mm_mv=10., resolution=200, grid=True,
                  grid_color=5, black_white=False, calibration_pulse=True,
                  image_noise_std=0., blur_radius=0., jpeg_quality=100, seed=0,
                  default_row_height_mm=30., save_image=None):
    """Render every page; return file paths and exact per-panel sample metadata.

    ``values`` and native labels remain canonical. Polarity affects only printed
    trace coordinates. Missing/short source intervals raise rather than pad.
    """
    values=np.asarray(values,dtype=np.float64)
    layout=_dictionary(layout)
    if values.ndim!=2 or values.shape[1]!=len(lead_names) or not np.isfinite(values).all():
        raise ValueError('A finite physical time-by-lead signal is required')
    if any(not np.isfinite(x) or x<=0 for x in (fs,speed_mm_s,gain_mm_mv,resolution)):
        raise ValueError('Clock, speed, gain and resolution must be positive')
    names={str(name).upper():i for i,name in enumerate(lead_names)}
    if len(names)!=len(lead_names):raise ValueError('Duplicate acquired lead names')
    if not layout.get('pages') or not layout.get('panels'):
        raise ValueError('A layout needs pages and panels')
    for page in layout['pages']:
        page=_dictionary(page)
        if any(not isinstance(page.get(key),int) or page[key]<1 for key in ('rows','columns')):
            raise ValueError('Page dimensions must be positive integers')
        if any(not np.isfinite(page.get(key,default)) or page.get(key,default)<0
               for key,default in (('margin_mm',10.),('column_gap_mm',5.))):
            raise ValueError('Page spacing must be finite and nonnegative')
        row_height=page.get('row_height_mm')
        if row_height is not None and (not np.isfinite(row_height) or row_height<=0):
            raise ValueError('Row height must be positive')
    occupied=set()
    for panel in layout['panels']:
        p=_dictionary(panel)
        if any(not isinstance(p.get(key),int) or p[key]<0 for key in ('page','row','column')):
            raise ValueError('Panel positions must be nonnegative integers')
        if p['page']>=len(layout['pages']):raise ValueError('Panel refers to absent page')
        if any(not isinstance(p.get(key,1),int) or p.get(key,1)<1 for key in ('row_span','column_span')):
            raise ValueError('Panel spans must be positive integers')
        page=_dictionary(layout['pages'][p['page']])
        if p['lead'].upper() not in names:raise ValueError(f"Source lacks declared lead: {p['lead']}")
        if p.get('polarity',1) not in (-1,1):raise ValueError('Invalid print polarity')
        if not np.isfinite(p['duration_s']) or not np.isfinite(p['start_s']) or p['duration_s']<=0 or p['start_s']<0:raise ValueError('Invalid panel time window')
        if round((p['start_s']+p['duration_s'])*fs)>len(values):
            raise ValueError('Declared panel extends beyond the acquired signal')
        for r in range(p['row'],p['row']+p.get('row_span',1)):
            for c in range(p['column'],p['column']+p.get('column_span',1)):
                if r>=page['rows'] or c>=page['columns'] or (p['page'],r,c) in occupied:
                    raise ValueError('Panels overlap or extend beyond page grid')
                occupied.add((p['page'],r,c))
    declared_duration=layout.get('record_duration_s')
    if declared_duration is not None and (not np.isfinite(declared_duration)
            or declared_duration<max(_dictionary(p)['start_s']+_dictionary(p)['duration_s'] for p in layout['panels'])
            or round(declared_duration*fs)>len(values)):
        raise ValueError('Declared record duration must be present in acquired signal')
    if grid_color not in range(1,6) or not np.isfinite(image_noise_std) or image_noise_std<0 \
            or not np.isfinite(blur_radius) or blur_radius<0 or not 1<=jpeg_quality<=100:
        raise ValueError('Invalid image augmentation or grid style')
    font_path=str(Path(__file__).parent/'Fonts/Times_New_Roman.ttf')
    style=dict(grid=grid,grid_color=grid_color,black_white=black_white,font_path=font_path)
    font=FontProperties(fname=font_path,size=9)
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    outputs=[]
    for page_number,page in enumerate(layout['pages']):
        page=_dictionary(page)
        geo=page_geometry(layout,page_number,speed_mm_s,calibration_pulse=calibration_pulse,
                          default_row_height_mm=default_row_height_mm)
        fig,ax,canvas,width_mm,height_mm,background=_canvas(geo,resolution,speed_mm_s,gain_mm_mv,style)
        leads=[]
        for p in geo['panels']:
            start=int(round(p['start_s']*fs));end=int(round((p['start_s']+p['duration_s'])*fs))
            if not 0<=start<end<=len(values):raise ValueError('Empty or unavailable panel samples')
            polarity=p.get('polarity',1)
            x=p['x_mm']+np.arange(end-start)/fs*speed_mm_s
            y=p['baseline_mm']-values[start:end,names[p['lead'].upper()]]*polarity*gain_mm_mv
            line,=ax.plot(x,y,linewidth=.75,color='black',zorder=2)
            label=('-' if polarity==-1 else '')+p['lead']
            text=ax.text(p['x_mm']+1.,p['baseline_mm']-4.,label,fontproperties=font,color='black',zorder=3)
            px=ax.transData.transform(np.column_stack((x,y)))
            px[:,1]=canvas.get_width_height()[1]-px[:,1]
            points=np.round(px[:,::-1],2)
            box=text.get_window_extent(renderer=canvas.get_renderer())
            text_points=np.array([[canvas.get_width_height()[1]-box.y1,box.x0],
                                  [canvas.get_width_height()[1]-box.y0,box.x1]])
            annotation=dict(lead_name=p['lead'],canonical_lead=p['lead'],printed_label=label,
                            polarity=polarity,page=page_number,row=p['row'],column=p['column'],
                            row_span=p.get('row_span',1),column_span=p.get('column_span',1),
                            kind=p.get('kind','lead'),start_sample=start,end_sample=end,
                            start_s=start/fs,duration_s=(end-start)/fs,
                            requested_start_s=p['start_s'],requested_duration_s=p['duration_s'],
                            baseline_mm=p['baseline_mm'],trace_origin_mm=p['x_mm'],
                            lead_bounding_box=_box(points),text_bounding_box=_box(text_points),
                            plotted_pixels=points.tolist())
            if calibration_pulse and p['column']==0:
                right=p['x_mm']-2.;left=right-geo['pulse_width_mm']
                pulse_x=[left,left,right,right];pulse_y=[p['baseline_mm'],p['baseline_mm']-gain_mm_mv,
                                                        p['baseline_mm']-gain_mm_mv,p['baseline_mm']]
                ax.plot(pulse_x,pulse_y,linewidth=.75,color='black',zorder=2)
                annotation['calibration_pulse_mm']=list(zip(pulse_x,pulse_y))
            leads.append(annotation)
        ax.text(geo['margin_mm'], height_mm-3., f'{speed_mm_s:g} mm/s     {gain_mm_mv:g} mm/mV',
                fontproperties=font, color='black', zorder=3)
        # Restore the static paper then draw all dynamic artists in original order.
        canvas.restore_region(background)
        for artist in sorted(list(ax.lines)+list(ax.texts),key=lambda item:item.zorder):
            ax.draw_artist(artist)
        rgba=np.asarray(canvas.buffer_rgba())
        pixels=Image.fromarray(rgba).convert('RGB')
        if image_noise_std:
            array=np.asarray(pixels,dtype=np.float32)
            for first in range(0,len(array),64):
                chunk=array[first:first+64];chunk+=rng.normal(0,image_noise_std,chunk.shape)
            np.clip(array,0,255,out=array);pixels=Image.fromarray(array.astype(np.uint8))
        if blur_radius:pixels=pixels.filter(ImageFilter.GaussianBlur(blur_radius))
        if jpeg_quality<100:
            import io
            compressed=io.BytesIO();pixels.save(compressed,format='JPEG',quality=jpeg_quality);compressed.seek(0)
            with Image.open(compressed) as image:pixels=image.convert('RGB')
        path=output_directory/f'{record_name}-page-{page_number:02}.png'
        if save_image is None:pixels.save(path)
        else:save_image(pixels,path)
        metadata=dict(schema='declarative-ecg-layout-v1',page=page_number,pages=len(layout['pages']),
                      sampling_frequency=fs,width=pixels.width,height=pixels.height,resolution=resolution,
                      paper_width_mm=width_mm,paper_height_mm=height_mm,pixels_per_mm=resolution/25.4,
                      paper_speed_mm_s=speed_mm_s,gain_mm_mv=gain_mm_mv,
                      x_grid=5*resolution/25.4,y_grid=5*resolution/25.4,
                      seconds_per_major_grid=5/speed_mm_s,mv_per_major_grid=5/gain_mm_mv,
                      gridlines=bool(grid),bw=bool(black_white),dc_pulse=bool(calibration_pulse),
                      number_of_columns_in_image=page['columns'],leads=leads,layout_spec=layout,
                      page_geometry={k:v for k,v in geo.items() if k!='panels'},
                      image_augmentation=dict(image_noise_std=image_noise_std,blur_radius=blur_radius,
                                              jpeg_quality=jpeg_quality,seed=seed))
        path.with_suffix('.json').write_text(json.dumps(metadata,separators=(',',':'))+'\n')
        outputs.append(str(path))
    return outputs
