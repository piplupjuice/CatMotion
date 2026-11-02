"""The machine: a two-axis gantry plotter with a pen that physically touches paper.

Why a controller and not a position actuator: a `<position>` actuator can only chase
a target, so a constant-velocity sweep leaves a steady-state lag of kv*v/kp - on this
machine that is a visible offset that bends at every direction change. Real plotters
feed the motion forward instead. Here the XY axes run a proportional law with the
reference velocity fed directly into the derivative term,

    f = kp*(x_ref - x) + kv*(v_ref - v)

which removes the steady-state lag a pure position actuator would leave behind. There
is no mass-times-acceleration term: the law above is the one that runs, and every
tracking and contact number this project reports was measured under it. The Z
axis is a plain position actuator, because a pen lift does not need tracking - it
needs a repeatable normal force on the paper, which is measured and reported.

Units: the model is in metres. One canvas unit (the artwork's [-1,1] square) is
SCALE metres, so a full drawing is 20 cm across - an honest desk plotter.
"""
import numpy as np
import mujoco

SCALE = 0.1          # metres per canvas unit
LIFT = 0.006         # metres the pen rises off the paper
PRESS = 0.0004       # extra travel past touch, so the ball loads the paper
TIP_R = 0.0008       # ballpoint radius
CARRIAGE_H = 0.12    # gantry height
DT = 0.0005          # physics timestep

_XML = """
<mujoco model="pen-plotter">
  <option timestep="{dt}" iterations="8" tolerance="1e-9" solver="Newton">
    <flag warmstart="enable"/>
  </option>

  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0.02" frictionloss="0.01"/>
  </default>

  <asset>
{asset}
  </asset>

  <worldbody>
    <light type="directional" dir="0 0 -1" pos="0 0 1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="4 4 0.05" pos="0 0 -0.01" rgba="0.085 0.085 0.095 1"/>

    <!-- the sheet. its top face is exactly z = 0, which is the canvas plane -->
    <body name="paper" pos="0 0 0">
      <geom name="sheet" type="box" size="{half} {half} 0.002" pos="0 0 -0.002"
            contype="1" conaffinity="1" friction="0.6 0.02 0.001"
            solref="0.004 1" solimp="0.9 0.95 0.001" rgba="0.92 0.92 0.90 1"/>
    </body>

    <!-- the fixed part of the machine: four posts and the two side rails the bridge
         rides. they sit outside the sheet in x, so a camera looking straight down
         never has a rail crossing the drawing where the drawing is. these geoms carry
         no mass, so they cannot change the dynamics the ink was measured against. -->
    <geom name="post_A" type="box" size="0.007 0.007 0.066" pos="-0.152 -0.132 0.066"
          mass="0" rgba="0.16 0.17 0.20 1"/>
    <geom name="post_B" type="box" size="0.007 0.007 0.066" pos="-0.152 0.132 0.066"
          mass="0" rgba="0.16 0.17 0.20 1"/>
    <geom name="post_C" type="box" size="0.007 0.007 0.066" pos="0.152 -0.132 0.066"
          mass="0" rgba="0.16 0.17 0.20 1"/>
    <geom name="post_D" type="box" size="0.007 0.007 0.066" pos="0.152 0.132 0.066"
          mass="0" rgba="0.16 0.17 0.20 1"/>
    <geom name="rail_L" type="capsule" fromto="-0.152 -0.140 0.126 -0.152 0.140 0.126"
          size="0.0035" mass="0" rgba="0.55 0.58 0.64 1"/>
    <geom name="rail_R" type="capsule" fromto="0.152 -0.140 0.126 0.152 0.140 0.126"
          size="0.0035" mass="0" rgba="0.55 0.58 0.64 1"/>

    <!-- a flatbed gantry: the bridge slides along y, the head rides the bridge in x.
         the nesting is the same two-axis positioning either way, but this way the
         only thing that ever crosses the paper is the machine doing its work. -->
    <body name="ycar" pos="0 0 {h}">
      <joint name="y" type="slide" axis="0 1 0" range="-{rng} {rng}"/>
      <geom name="gy" type="box" size="0.150 0.012 0.008" mass="{my}" rgba="0.4 0.45 0.55 1"/>
      <body name="xcar">
        <joint name="x" type="slide" axis="1 0 0" range="-{rng} {rng}"/>
        <geom name="gx" type="box" size="0.022 0.026 0.014" mass="{mx}" rgba="0.35 0.4 0.5 1"/>
        <body name="pen" pos="0 0 -{d}">
          <joint name="z" type="slide" axis="0 0 -1" range="0 {zmax}"/>
          <geom name="barrel" type="capsule" fromto="0 0 0.03 0 0 -0.01" size="0.004"
                mass="{mz}" rgba="0.75 0.2 0.25 1"/>
          <geom name="tip" type="sphere" size="{tip}" pos="0 0 -0.014" mass="0.004"
                contype="1" conaffinity="1" rgba="0.1 0.1 0.12 1"/>
        </body>
      </body>
    </body>
    {world}
  </worldbody>

  <actuator>
    <motor name="fx" joint="x" ctrlrange="-60 60" gear="1"/>
    <motor name="fy" joint="y" ctrlrange="-60 60" gear="1"/>
    <position name="pz" joint="z" kp="800" kv="14" ctrlrange="-0.004 {zmax}"
             forcerange="-20 20"/>
  </actuator>
</mujoco>
"""


def build_xml(scale=SCALE, asset="", world=""):
    """Fill in the geometry so the tip just touches the sheet at q_z = LIFT.

    `asset` and `world` let the renderer bolt its ink surface and lights onto the
    same model the physics ran on, instead of maintaining a second XML that can
    drift away from the machine that made the drawing.
    """
    half = 1.15 * scale
    rng = 1.35 * scale
    # tip centre sits at (h - d - q_z - 0.014); at q_z = 0 it must be LIFT above z=0
    d = CARRIAGE_H - TIP_R - LIFT - 0.014
    return _XML.format(dt=DT, half=half, h=CARRIAGE_H, rng=rng, d=d, tip=TIP_R,
                        zmax=LIFT + PRESS + 0.002, mx=0.60, my=0.45, mz=0.05,
                        asset=asset, world=world)


class Plotter:
    """Gantry plotter driven by a feed-forward XY controller."""

    def __init__(self, scale=SCALE, bandwidth=45.0, zeta=1.0):
        self.scale = float(scale)
        self.model = mujoco.MjModel.from_xml_string(build_xml(scale))
        self.data = mujoco.MjData(self.model)
        self.ix = self.model.joint("x").qposadr[0]
        self.iy = self.model.joint("y").qposadr[0]
        self.iz = self.model.joint("z").qposadr[0]
        self.ux = self.model.joint("x").dofadr[0]
        self.uy = self.model.joint("y").dofadr[0]
        self.cx, self.cy = self.model.actuator("fx").id, self.model.actuator("fy").id
        self.cz = self.model.actuator("pz").id
        # subtree masses are the inertias each axis actually has to accelerate
        self.mm = tuple(float(self.model.body_subtreemass[self.model.body(b).id])
                        for b in ("xcar", "ycar"))
        w = 2 * np.pi * bandwidth
        self.kp = np.array([w ** 2 * m for m in self.mm])
        self.kv = np.array([2 * zeta * w * m for m in self.mm])
        # gravity droops the pen by m*g/kp below its commanded height, which reads as
        # a pen that never fully lifts. offsetting the setpoint cancels it exactly,
        # the same bias a real lift solenoid's spring provides.
        kp_z = float(self.model.actuator_gainprm[self.cz][0])
        m_pen = float(self.model.body_subtreemass[self.model.body("pen").id])
        self.z_ff = m_pen * abs(self.model.opt.gravity[2]) / kp_z
        mujoco.mj_forward(self.model, self.data)

    # -- reference ----------------------------------------------------------
    def reference(self, pr, fps=60.0):
        """Commanded xy (metres) at one entry per program frame, plus its velocity.

        The velocity is a central difference of the same fine polyline the ink is
        drawn from, so the feed-forward term is the motion the pen is genuinely asked
        to make rather than a smoothed guess.
        """
        xy = np.asarray(pr["xy"], np.float64) * self.scale
        v = np.gradient(xy, axis=0) * fps
        return {"xy": xy, "v": v, "pen": np.asarray(pr["pen"], np.float64),
                "draw": np.asarray(pr["draw"], bool), "dt_frame": 1.0 / fps}

    def pen_ctrl(self, p):
        """Joint travel for a pen command: 0 is airborne, 1 is on the paper.

        Gravity pulls the pen below its setpoint by m*g/kp, so the target is biased
        up by that much and the achieved height is the one that was asked for.
        """
        return (LIFT + PRESS) * float(p) - self.z_ff

    # -- simulation ---------------------------------------------------------
    def run(self, pr, fps=60.0, sub=None):
        """Simulate the whole program; return achieved xy in canvas units."""
        ref = self.reference(pr, fps)
        T = len(ref["xy"])
        n_sub = int(round((1.0 / fps) / DT)) if sub is None else int(sub)
        q, qd = self.data.qpos, self.data.qvel
        # the carriage starts parked where the program starts, not at the origin:
        # otherwise frame 0 is a step command and the first thing the machine does
        # is saturate its own actuators
        q[:] = 0.0
        q[self.ix], q[self.iy] = ref["xy"][0]
        self.data.ctrl[self.cz] = self.pen_ctrl(0.0)
        mujoco.mj_forward(self.model, self.data)

        nxt = np.concatenate([ref["xy"][1:], ref["xy"][-1:]], axis=0)
        got = np.zeros((T, 2))
        pen_got = np.zeros(T)
        force = np.zeros(T)
        err = np.zeros(T)
        for i in range(T):
            x0, x1 = ref["xy"][i], nxt[i]
            vd = ref["v"][i]
            fmax = 0.0
            self.data.ctrl[self.cz] = self.pen_ctrl(ref["pen"][i])
            for s in range(n_sub):
                # the setpoint sweeps across the frame, so the feed-forward velocity
                # and the position target describe the same motion. holding the
                # target still while feeding forward its velocity biases the pen
                # backwards by kv*v/kp - about 0.2 mm here, visible at 1080p
                xd = x0 + (x1 - x0) * ((s + 1) / float(n_sub))
                e = xd - np.array([q[self.ix], q[self.iy]])
                ev = vd - np.array([qd[self.ux], qd[self.uy]])
                f = self.kp * e + self.kv * ev
                self.data.ctrl[self.cx] = np.clip(f[0], -60, 60)
                self.data.ctrl[self.cy] = np.clip(f[1], -60, 60)
                mujoco.mj_step(self.model, self.data)
                # a frame lays ink if the ball touched the paper at any instant in
                # it; sampling only the last instant catches the pen mid-oscillation
                # and reports a gap that is not on the page
                if self.data.ncon:
                    fmax = max(fmax, self.normal_force())
            got[i] = np.array([q[self.ix], q[self.iy]]) / self.scale
            # measured at the end of the frame, against where the frame was heading.
            # scoring it against the frame's start would report one frame of travel
            # as if it were servo error
            err[i] = float(np.linalg.norm(x1 - got[i] * self.scale)) / self.scale
            # 1.0 means the ball is on the paper (touching or pressed), 0 airborne
            pen_got[i] = float(np.clip(q[self.iz] / LIFT, 0.0, 1.0))
            force[i] = fmax
        return {"xy": got, "pen": pen_got, "force": force, "err": err,
                "cmd": np.asarray(pr["xy"], np.float64), "draw": ref["draw"],
                "fps": fps, "T": T}

    def normal_force(self):
        """Total contact normal force on the tip, newtons - zero when it is airborne."""
        d, m = self.data, self.model
        gid = m.geom("tip").id
        f = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 == gid or c.geom2 == gid:
                n = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, n)
                f += abs(n[0])
        return float(f)


if __name__ == "__main__":
    import art
    import planner

    st = art.fit_canvas(art.cat_strokes())
    pr = planner.plan(st, first="head")
    rob = Plotter()
    out = rob.run(pr)

    ink = out["draw"]
    travel = np.asarray(pr["role"]) == planner.R_TRAVEL
    e_ink = out["err"][ink]
    assert np.isfinite(out["xy"]).all() and np.isfinite(out["force"]).all(), "sim diverged"
    assert np.abs(out["xy"]).max() < 1.35, "carriage left the sheet"
    # tracking: the pen must land where the program says, in canvas units
    assert e_ink.max() < 0.005, "worst ink-tracking error %.5f units" % e_ink.max()
    assert np.sqrt((e_ink ** 2).mean()) < 0.0015, "rms ink error %.5f" % np.sqrt((e_ink ** 2).mean())
    # contact is real: force exactly where the program draws ink, and nowhere else
    assert (out["force"][ink] > 0.01).mean() > 0.98, "the pen was not on the paper"
    assert out["force"][travel].max() < 1e-6, "the pen dragged while flying"
    assert out["pen"][ink].mean() > 0.98, "pen never reached the paper"
    assert out["pen"][travel].max() < 0.05, "pen not lifted between strokes"

    print("ROBOT PASS  T=%d  ink err rms %.5f max %.5f units  normal force %.3f N"
          % (out["T"], float(np.sqrt((e_ink ** 2).mean())), float(e_ink.max()),
             float(out["force"][ink].mean())))
    print("        flight err max %.5f  lift %.2f mm, press to %.2f mm, tip r %.3f mm"
          % (float(out["err"][travel].max()), LIFT * 1000, (LIFT + PRESS) * 1000,
             TIP_R * 1000))
    print("        physics dt %.4f s  control frame %.4f s (%.0f Hz)  %d substeps"
          % (DT, 1.0 / out["fps"], out["fps"], int(round(1.0 / out["fps"] / DT))))
