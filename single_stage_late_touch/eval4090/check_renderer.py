"""Small MuJoCo/NVIDIA render check, without running a policy rollout."""
import ctypes
import json
import mujoco
import numpy as np

assert mujoco.__version__ == '3.4.0', mujoco.__version__
model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="128" offheight="96"/></visual>
<worldbody><light pos="0 0 2"/><geom type="plane" size="1 1 .1"/>
<geom type="sphere" size=".1" pos="0 0 .2" rgba="1 0 0 1"/></worldbody></mujoco>''')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=96, width=128) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    renderer._gl_context.make_current()
    gl = ctypes.CDLL('libGL.so.1')
    gl.glGetString.restype = ctypes.c_char_p
    vendor = gl.glGetString(0x1F00).decode()
    device = gl.glGetString(0x1F01).decode()
    assert 'NVIDIA' in vendor and '4090' in device, (vendor, device)
    assert np.isfinite(frame).all() and frame.std() > 0
    print(json.dumps({'mujoco': mujoco.__version__, 'vendor': vendor, 'renderer': device,
                      'shape': frame.shape, 'pixel_std': float(frame.std()), 'passed': True}))
