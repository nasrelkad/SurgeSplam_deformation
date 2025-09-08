from plyfile import PlyData, PlyElement
import numpy as np

old_ply = 'gaussians.ply'
new_ply = 'gaussians_supersplat.ply'

ply = PlyData.read(old_ply)
v = ply['vertex'].data

# Create a new structured array with correct field order
new_v = np.empty(len(v), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('nw', 'f4'),
    ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4'),
    ('opacity', 'f4'),
    ('f_dc_0', 'u1'), ('f_dc_1', 'u1'), ('f_dc_2', 'u1')
])

for field in ['x','y','z','nx','ny','nz','nw','sx','sy','sz']:
    new_v[field] = v[field]

new_v['opacity'] = v['opacity']
new_v['f_dc_0'] = v['f_dc_0'] if 'f_dc_0' in v.dtype.names else v['red']
new_v['f_dc_1'] = v['f_dc_1'] if 'f_dc_1' in v.dtype.names else v['green']
new_v['f_dc_2'] = v['f_dc_2'] if 'f_dc_2' in v.dtype.names else v['blue']

PlyData([PlyElement.describe(new_v, 'vertex')], text=True).write(new_ply)
print(f"[OK] Saved Supersplat-compatible PLY: {new_ply}")
