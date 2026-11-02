"""The film. Ink is a texture painted onto the paper, not a line drawn on screen.

That single choice is what makes the footage honest and what makes every camera
angle work. The pen's *measured* contact positions are splatted into a 2D texture
mounted on the sheet, so the drawing becomes a property of the page inside the 3D
scene. A top-down flat lay, a low dolly, a macro on the ballpoint: all of them see
the same ink at the same place on the paper, with real perspective, real occlusion
by the carriage, and nothing to register in screen space.

Nothing here is assumed. The map from canvas coordinates to texture texels is
*measured* at build time with an emissive probe placed in world coordinates, then
verified against held-out points and against the finished drawing as a whole
before a single film frame is written. If the registration is wrong this module
fails loudly instead of drawing a pretty lie.
"""
import os
import sys

# mujoco binds its GL backend at import time, so this has to run before that import:
# windows draws through wgl, linux (the GPU box) has only egl.
os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "egl")

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

import robot

# generated pictures belong to the project, not to whichever directory the script
# happened to be started from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "design_out")

N_TEX = 1024             # ink texture, square
PAPER_RGB = (235, 233, 227)
# Two flag arrays, one name. `MjvOption.flags` (31 long) is indexed by mjtVisFlag and
# decides what is *drawn*; `mjvScene.flags` (11 long) is indexed by mjtRndFlag and
# decides how it is *rendered*. Shadows live in the second. See `set_shadow`.
RND_SHADOW = int(mujoco.mjtRndFlag.mjRND_SHADOW)
INK_RGB = (24, 27, 41)   # ballpoint blue-black
MARK_RGB = (255, 0, 255)  # painted calibration marker
F_MIN = 0.01             # newtons: a frame laid ink if the tip pressed this hard
LINE_W = 0.016           # canvas units; the tip is 1.6 mm across
# the calibration gate: the ink texture has to map once across the sheet, and "once" is
# allowed a fifth of slack either way before the camera is refused. Named so the
# self-test can print the gate it applied instead of leaving the reader to trust it.
TPU_LO, TPU_HI = 0.8, 1.25
# the two gates the film has to clear before it is allowed to be watched: every stroke
# found within REG_GATE of where the pen put it, and the finished page's ink centroid
# within CENTROID_GATE of the program's. Printed by the self-test, so the report can
# quote the bar and not only the score.
REG_GATE, CENTROID_GATE = 6.0, 12.0
# how much of its light the shadow is allowed to take off the worst-lit strip of the
# sheet. A gate that is only ever asserted is a gate nobody can quote, so the self-test
# prints it next to the measurement.
SHADOW_GATE = 0.30
# the shadow rig, both set before the Renderer is built because MuJoCo allocates the
# depth texture from them. These are the ceiling of this GL backend, not a preference:
# shadowsize 4096 and any shadowscale but 1.0 both return a black page here, so the
# gantry shadow keeps a stepped edge at macro focal lengths and the films say so.
SHADOW_SIZE, SHADOW_SCALE = 2048, 1.0
MARK_R = 14.0            # texels: a calibration marker has to survive texture filtering
# Calibration markers live inside the drawing area. The gantry's x-carriage rides at
# mid-height and its body covers the page's outer margin forever - that is a fact
# about the machine, not an error in the map, but a marker half-hidden behind it gives
# a half-disc and a biased centroid.
VERIFY_POINTS = [(0.62, -0.44), (-0.78, 0.31), (0.27, 0.76), (-0.55, -0.70), (0.74, 0.05)]


# --------------------------------------------------------------------- surface
def mark_weight(img):
    """Per-pixel magenta-ness of a rendered frame; zero outside the marker hue.

    The gate is a ratio, not an absolute level. The same marker renders at a
    different brightness under a different rig, and an absolute threshold quietly
    stops finding ink the moment the exposure moves: a marker's own pixels sit at
    about 1.0-1.2 times their luminance of blue-or-red over green, while cool
    shadowed paper reaches only a few counts, well under a tenth of its luminance.
    """
    v = np.asarray(img, np.float32)
    mn = np.minimum(v[:, :, 0], v[:, :, 2])
    w = mn - v[:, :, 1]
    r = w / (v.mean(axis=2) + 1.0)
    return np.where((mn > 45.0) & (r > 0.16), np.clip(w, 0.0, None), 0.0)


def ink_xml(n, scale):
    """Texture, material, ink plane, calibration probe and a key light."""
    half = 1.15 * scale
    asset = """
    <texture name="ink" type="2d" builtin="flat" width="{n}" height="{n}"
             rgb1="0.92 0.91 0.89" rgb2="0.92 0.91 0.89"/>
    <material name="inkmat" texture="ink" texuniform="false" rgba="1 1 1 1"
              specular="0.06" shininess="0.03" emission="0.03"/>
    <material name="probem" rgba="0 1 0.42 1" emission="1"/>
    """.format(n=n)
    world = """
    <geom name="ink" type="plane" size="{h} {h} 0.001" pos="0 0 0.00012"
          material="inkmat" contype="0" conaffinity="0"/>
    <geom name="probe" type="sphere" size="0.0022" pos="0 0 0.02"
          material="probem" contype="0" conaffinity="0"/>
    <light type="directional" dir="0.34 -0.46 -0.82" pos="0.55 -0.75 1.2"
           diffuse="0.44 0.44 0.50" specular="0.10 0.10 0.10"/>
    <light type="directional" dir="-0.62 0.34 -0.42" pos="-0.9 0.5 0.8"
           diffuse="0.17 0.18 0.24" specular="0.06 0.06 0.08"/>
    """.format(h=half)
    return asset, world


class Ink:
    """The page: an RGBA-free texture buffer plus the measured rule for where ink goes."""

    def __init__(self, n=N_TEX, seed=3):
        self.n = int(n)
        rng = np.random.default_rng(seed)
        g = rng.normal(0.0, 5.0, (self.n, self.n, 1))
        self.buf = np.clip(np.asarray(PAPER_RGB, np.float32) + g, 0, 255).astype(np.uint8)
        self.base = self.buf.copy()
        self.M = None      # canvas (x, y, 1) -> buffer (row, col)
        self.tpu = None    # texels per canvas unit

    def clear(self):
        self.buf[:] = self.base

    def set_map(self, M):
        self.M = np.asarray(M, np.float64)
        self.tpu = float(np.sqrt(abs(np.linalg.det(self.M[:, :2]))))

    def cell(self, xy):
        s = self.M @ np.array([float(xy[0]), float(xy[1]), 1.0])
        return s[0], s[1]

    def splat(self, rc, radius, rgb=INK_RGB):
        r, c, n = float(rc[0]), float(rc[1]), self.n
        rr = int(np.ceil(radius)) + 1
        r0, r1 = max(0, int(r) - rr), min(n, int(r) + rr + 2)
        c0, c1 = max(0, int(c) - rr), min(n, int(c) + rr + 2)
        if r1 <= r0 or c1 <= c0:
            return 0
        ys, xs = np.mgrid[r0:r1, c0:c1]
        w = np.clip(radius + 0.5 - np.hypot(ys - r, xs - c), 0.0, 1.0)
        patch = self.buf[r0:r1, c0:c1].astype(np.float32)
        tgt = np.asarray(rgb, np.float32)
        self.buf[r0:r1, c0:c1] = np.clip(patch * (1.0 - w[..., None]) + tgt * w[..., None],
                                         0.0, 255.0).astype(np.uint8)
        return int((w > 0.5).sum())

    def dot(self, xy, width, rgb=INK_RGB):
        return self.splat(self.cell(xy), 0.5 * width * self.tpu, rgb)

    def line(self, a, b, width, rgb=INK_RGB):
        """Stroke a segment; dots are spaced at a quarter of the line width."""
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        gap = max(0.25 * width, 1e-9)
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / gap)))
        for t in np.arange(k + 1) / k:
            self.dot(a + t * (b - a), width, rgb)

    def area(self):
        """Ink coverage in canvas units squared - the page's own bookkeeping."""
        dark = (self.buf.min(axis=2) < 120)
        return int(dark.sum()) / (self.tpu ** 2)


# ------------------------------------------------------------------------ scene
def _dist(scene, half_view):
    """Camera distance that frames `half_view` metres of subject, vertical."""
    return half_view / np.tan(np.deg2rad(scene.fovy) / 2.0)


class Scene:
    """Model + renderer + the calibration that ties canvas coordinates to the page."""

    def __init__(self, scale=robot.SCALE, width=1920, height=1080, n_tex=N_TEX,
                 calibrate=True):
        self.scale = float(scale)
        self.width, self.height = int(width), int(height)
        self.ink = Ink(n_tex)
        asset, world = ink_xml(n_tex, self.scale)
        self.model = mujoco.MjModel.from_xml_string(
            robot.build_xml(self.scale, asset, world))
        self.data = mujoco.MjData(self.model)
        self.tex = self.model.texture("ink").id
        self.adr = int(self.model.tex_adr[self.tex])
        self.probe_geom = self.model.geom("probe").id
        assert int(self.model.tex_nchannel[self.tex]) == 3, "ink texture is not RGB"
        assert (int(self.model.tex_width[self.tex]), int(self.model.tex_height[self.tex])) \
            == (n_tex, n_tex), "the texture was not built at the size the page expects"
        self.fovy = float(self.model.vis.global_.fovy)

        hl = self.model.vis.headlight
        # the headlight is the only light MuJoCo does not shadow, so it is what fills
        # the band the bridge drops across the page. with a dim headlight the shadow is
        # a grey stripe laid over the drawing and the film reads as a rendering bug;
        # these values keep the shadowed paper above half the lit paper, which is what
        # page_contrast below refuses to accept less than.
        hl.ambient = (0.16, 0.16, 0.19)
        hl.diffuse = (0.30, 0.30, 0.33)
        hl.specular = (0.03, 0.03, 0.03)
        # the machine model carries a light straight down, and a light straight down puts
        # the bridge's shadow exactly behind the bridge: a hard grey stripe the width of
        # the gantry drawn across the page, which is the one artefact that makes a plotter
        # film look broken. it stays, because it is what makes the flat lay even, but at a
        # quarter - the two angled lights shape the machine and fill the band with light
        # that arrives from somewhere the metal is not.
        down = int(np.argmin(self.model.light_dir[:, 2]))
        assert np.allclose(self.model.light_dir[down], (0.0, 0.0, -1.0), atol=0.05), \
            "the lowest-pointing light is not the vertical one the rig assumes"
        self.model.light_diffuse[down] *= 0.25
        # three lights on one sheet of paper is enough to clip it to pure white, and a
        # white page has no texture and no film. the illuminance on the sheet is the
        # SUM of the lights, so the rig is scaled on its total, not its average: an
        # average lets a dim fill light brighten the key by the same factor it dilutes
        # the mean, which is how adding one fill light burned the page out.
        lit = self.model.light_diffuse
        lum = float(np.maximum(lit, 0.0).sum())
        if lum > 0:
            lit[:] *= min(1.0, 0.95 / lum)
        self.model.vis.quality.shadowsize = SHADOW_SIZE
        # how much of the scene the shadow map is spread over. the map is fixed at the
        # size the driver will actually allocate, so the only way to spend its pixels on
        # the paper rather than on the room is to shrink the region - and a region that
        # is too big is what stair-cases the edge of the gantry shadow across the page.
        self.model.vis.map.shadowscale = SHADOW_SCALE
        g = self.model.vis.global_
        g.offwidth = max(int(g.offwidth), self.width)
        g.offheight = max(int(g.offheight), self.height)
        self.r = mujoco.Renderer(self.model, self.height, self.width)
        # the option the renderer would use anyway, copied rather than rebuilt: a
        # hand-made MjvOption comes with the debug decorations switched on - contact
        # points among them - and an all-zero one leaves the static sheet undrawn.
        # whatever this rig's own default is, that is what the picture was made with.
        self.opt = mujoco.MjvOption()
        self.opt.flags[:] = self.r._scene_option.flags
        # shadows off until `calibrate` has finished: the calibration reads small bright
        # markers off the paper, and a gantry shadow across them is the same failure mode
        # as an over-lit page. calibrate turns it on at the end and proves it works.
        self.set_shadow(False)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.M = None
        self.top_orient = None
        self.calib_err_px = None
        self.park_y = -1.28
        if calibrate:
            self.calibrate()

    # -- gl -----------------------------------------------------------------
    def upload(self):
        m = self.model
        n = int(m.tex_width[self.tex]) * int(m.tex_height[self.tex]) * int(m.tex_nchannel[self.tex])
        m.tex_data[self.adr:self.adr + n] = self.ink.buf.ravel()
        mujoco.mjr_uploadTexture(m, self.r._mjr_context, self.tex)

    def apply_cam(self, c):
        self.cam.orthographic = bool(c["orthographic"])
        self.cam.elevation = float(c["elevation"])
        self.cam.azimuth = float(c["azimuth"])
        self.cam.distance = float(c["distance"])
        self.cam.lookat[:] = c["lookat"]

    def render(self):
        # mjv_updateScene reads cached body/geom poses, so a qpos or geom_pos change
        # is invisible until the model is forwarded.
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        return self.r.render()

    def camera_pos(self):
        """The current camera's world position, as MuJoCo placed it (see `park`)."""
        self.render()
        return np.array(self.r.scene.camera[0].pos)

    def set_shadow(self, on):
        # shadows are a flag on the *scene*, indexed by mjtRndFlag, not one on the
        # option, indexed by mjtVisFlag. the two arrays are different lengths (11 and
        # 31) and the shadow's index in the short one - 0 - is also a legal index in
        # the long one, where it means mjVIS_COM: writing it there hangs a grey ball
        # the size of a fist off the carriage and then measures that ball as the
        # shadow it was supposed to be. the read-back is the point - a shadow that is
        # not really on is a lie about the picture, and the picture is the deliverable.
        f = self.r.scene.flags
        f[RND_SHADOW] = bool(on)
        assert bool(self.r.scene.flags[RND_SHADOW]) == bool(on), "the shadow flag did not take"

    def park(self):
        """Carriage hard into a corner, pen lifted, probe off the page.

        The bridge spans the whole gantry and rides a decimetre above the sheet, so
        parked on the camera's side of the page it films a stripe straight across the
        finished drawing. `park_y` is the far side, measured from the hero camera.
        """
        q = self.data.qpos
        q[:] = 0.0
        q[self.model.joint("x").qposadr[0]] = 1.32 * self.scale
        q[self.model.joint("y").qposadr[0]] = self.park_y * self.scale
        self.set_probe((9.0, 9.0))

    def set_carriage(self, xy, qz=None):
        q = self.data.qpos
        q[self.model.joint("x").qposadr[0]] = float(xy[0]) * self.scale
        q[self.model.joint("y").qposadr[0]] = float(xy[1]) * self.scale
        q[self.model.joint("z").qposadr[0]] = (robot.LIFT if qz is None else float(qz))

    def set_probe(self, xy):
        p = self.model.geom_pos
        p[self.probe_geom] = (float(xy[0]) * self.scale, float(xy[1]) * self.scale, 0.02)

    # -- detection ----------------------------------------------------------
    @staticmethod
    def _centroid(img, mask):
        m = mask(img.astype(np.int32))
        n = int(m.sum())
        if n < 8:
            return None
        ys, xs = np.nonzero(m)
        return (float(xs.mean()), float(ys.mean()), n)

    def find_probe(self, img):
        return self._centroid(img, lambda v: (v[:, :, 1] > v[:, :, 0] + 50)
                              & (v[:, :, 1] > v[:, :, 2] + 30) & (v[:, :, 1] > 140))

    def find_mark(self, img):
        """Magenta-ness weighted centroid.

        A thresholded disc is only as good as its edge, and its edge is a blend with
        the page, so a hard mask lets the local paper gradient move the centroid by a
        few pixels. Weighting by how magenta each pixel is makes the estimate depend
        on the marker's core, where the signal is, rather than on its fringe.
        """
        # the hue gate lives in mark_weight, in one place with the weight
        w = mark_weight(img)
        ys, xs = np.nonzero(w)
        if len(xs) < 25:
            return None
        ww = w[ys, xs]
        s = float(ww.sum())
        return (float((xs * ww).sum() / s), float((ys * ww).sum() / s), len(xs))

    def mark_stats(self, img):
        """How close the image came to containing a marker, for a failure message."""
        w = mark_weight(img)
        k = int(np.argmax(w))
        return "best magenta weight %d at (%d, %d), %d px detected" % (
            w.max(), k % img.shape[1], k // img.shape[1], int((w > 0).sum()))

    def find_ink(self, img):
        """Inkness-weighted centroid of the drawing, inside the canvas rectangle.

        Measured as a difference from the blank page captured at calibration time,
        not as an absolute level: the page has a luminance gradient of its own, and a
        fixed threshold would read that gradient as ink and bias the centroid. The
        baseline is the top-down view, so this is a top-down measurement.
        """
        rect = self._canvas_rect()
        lum = img.astype(np.float32).mean(axis=2)
        ref = getattr(self, "baseline", None)
        if ref is None:
            ref = np.full_like(lum, float(np.median(lum[rect])))
        w = np.clip((ref - lum) / np.maximum(0.7 * ref, 1.0), 0.0, 1.0)
        w[~rect] = 0.0
        s = float(w.sum())
        if s < 1.0:
            return None
        ys, xs = np.nonzero(w)
        ww = w[ys, xs]
        return (float((xs * ww).sum() / s), float((ys * ww).sum() / s), s)

    def _canvas_rect(self):
        h, w = self.height, self.width
        pts = np.array([self.pixel_of((x, y)) for x in (-1.02, 1.02) for y in (-1.02, 1.02)])
        m = np.zeros((h, w), bool)
        c0, c1 = int(max(0, np.floor(pts[:, 0].min()))), int(min(w - 1, np.ceil(pts[:, 0].max())))
        r0, r1 = int(max(0, np.floor(pts[:, 1].min()))), int(min(h - 1, np.ceil(pts[:, 1].max())))
        m[r0:r1 + 1, c0:c1 + 1] = True
        return m

    def pixel_of(self, xy):
        p = self.pixel_from_canvas @ np.array([float(xy[0]), float(xy[1]), 1.0])
        return p

    # -- calibration --------------------------------------------------------
    def _top_cam(self, orient, half_view=None):
        hv = 1.24 * 1.15 * self.scale if half_view is None else half_view
        return {"orthographic": True, "elevation": orient[0], "azimuth": orient[1],
                "distance": _dist(self, hv), "lookat": (0.0, 0.0, 0.0)}

    def _probe_frame(self, xy):
        self.set_probe(xy)
        mujoco.mj_forward(self.model, self.data)
        return self.render()

    def calibrate(self):
        """Measure canvas->pixel and texel->pixel, compose them, then prove the result.

        The top-down orientation is chosen by search rather than by argument: eight
        candidates are tried and the one that puts canvas +x to the right and +y
        upward is kept, because MuJoCo's camera angles are not something to guess at.
        """
        self.park()
        best = None
        for el in (90.0, -90.0):
            for az in (0.0, 90.0, 180.0, -90.0):
                self.apply_cam(self._top_cam((el, az)))
                self.ink.clear()
                self.upload()
                p0 = self.find_probe(self._probe_frame((0.0, 0.0)))
                px = self.find_probe(self._probe_frame((0.7, 0.0)))
                py = self.find_probe(self._probe_frame((0.0, 0.7)))
                if p0 is None or px is None or py is None:
                    continue
                ax = (np.array(px[:2]) - np.array(p0[:2])) / 0.7
                ay = (np.array(py[:2]) - np.array(p0[:2])) / 0.7
                s = 0.5 * (np.linalg.norm(ax) + np.linalg.norm(ay))
                if s < 40.0:
                    continue          # the probe never moved: the model ignored it
                err = (abs(ax[0] - s) + abs(ax[1]) + abs(ay[0]) + abs(ay[1] + s)) / s
                if best is None or err < best[0]:
                    best = (err, (el, az), np.array(p0[:2]), ax, ay)
        assert best is not None, "no top-down orientation showed the probe at all"
        err, orient, p0, ax, ay = best
        assert err < 0.08, ("top-down camera is not axis aligned (score %.3f)" % err)
        self.top_orient = orient
        # which side of the sheet the hero shot sits on decides where the parked
        # bridge gets out of the way. read the camera's position back from MuJoCo
        # instead of working it out from its angles: elevation is negative above the
        # subject and azimuth 90 is not the axis a viewer expects.
        self.apply_cam(camera("hero", 0.5, self))
        self.park_y = 1.34 if self.camera_pos()[1] < 0.0 else -1.34
        # the search loop leaves whichever orientation it tried last switched on. the
        # texel fit below has to be measured with the camera the film will actually
        # use, or the page is registered to a view that is rotated 180 degrees from it.
        self.apply_cam(self._top_cam(orient))
        # the registration measurements are geometry measurements. a marker that
        # happens to sit in the parked carriage's shadow has its magenta weight eaten
        # from one side, which reads as a position error the map never made.
        self.set_shadow(False)
        A = np.array([[ax[0], ay[0], p0[0]], [ax[1], ay[1], p0[1]]], np.float64)

        pts, pix, ns = [], [], []
        for (fr, fc) in [(0.15, 0.20), (0.85, 0.26), (0.24, 0.85), (0.83, 0.81)]:
            self.ink.clear()
            self.ink.splat((fr * self.ink.n, fc * self.ink.n), MARK_R, MARK_RGB)
            self.upload()
            m = self.find_mark(self.render())
            assert m is not None, "painted marker never rendered: texture is not on the sheet"
            pts.append((fr * self.ink.n, fc * self.ink.n))
            pix.append(m[:2])
            ns.append(m[2])
        assert min(ns) > 0.6 * max(ns), ("a fit marker was occluded (%s px)" % ns)
        X = np.c_[np.array(pts), np.ones(4)]
        sol, _, _, _ = np.linalg.lstsq(X, np.array(pix) - p0, rcond=None)
        # the lstsq maps texel -> pixel offset, so undo the offset and invert the 2x2
        A2 = sol[:2, :].T                            # (row, col) -> (x, y)
        t = sol[2, :] + p0
        G = np.linalg.inv(A2)
        texel_of_pixel = np.hstack([G, (-G @ t)[:, None]])
        A3 = np.vstack([A, np.array([0.0, 0.0, 1.0])])   # canvas -> pixel, homogeneous
        self.M = texel_of_pixel @ A3                     # canvas -> texel
        self.pixel_from_canvas = A
        self.ink.set_map(self.M)

        expect = self.ink.n / (2.0 * 1.15)     # the sheet is 2.3 canvas units across
        ratio = self.ink.tpu / expect
        assert TPU_LO < ratio < TPU_HI, ("texture does not map once across the sheet (tpu %.1f,"
                                         " expected %.1f)" % (self.ink.tpu, expect))
        worst = self.verify()
        self.park()
        self.set_shadow(False)
        med, clip, empty = self._expose()
        self.set_shadow(True)
        self.baseline = empty.astype(np.float32).mean(axis=2)
        frac, depth, iou = self.verify_shadow()
        print("         shadow covers %.1f%% of the paper at %.0f levels dark, "
              "and %.0f%% of it is where the last shadow was"
              % (100.0 * frac, depth, 100.0 * iou))
        assert frac > 0.02, ("the shadow flag darkens %.1f%% of the paper: the rig is "
                             "rendering without shadows" % (100.0 * frac))
        assert depth > 15.0, "the shadow is only %.0f levels deep: that is a tint, not a shadow" % depth
        assert iou < 0.75, ("the darkened paper is %.0f%% the same at both carriage "
                            "positions: that is not a shadow following the metal" % (100.0 * iou))
        print("CALIBRATE  %.0f texels/unit  top-down el=%+.0f az=%+.0f  scale %.3f of sheet"
              " (gate %.2f to %.2f)"
              % (self.ink.tpu, orient[0], orient[1], ratio, TPU_LO, TPU_HI))
        print("         held-out marker registration %.2f px worst of %d"
              % (worst, len(VERIFY_POINTS)))
        print("         paper median %d, clipped %.3f" % (med, clip))
        assert 120 < med < 250, "the page is not exposed (median %d)" % med
        assert clip < 0.30, "the page is burned out: %.1f%% of it is pure white" % (100 * clip)
        self.calib_err_px = worst

    def _scale_lights(self, k):
        self.model.light_diffuse[:] *= k
        self.model.light_specular[:] *= k
        mujoco.mj_forward(self.model, self.data)

    def _expose(self, want=195.0, band=(170, 225), iters=5):
        """Scale the rig until the blank page sits inside the wanted luminance band.

        What a light coefficient is worth in pixels depends on the paper albedo, the
        headlight, the specular lobe and how many lights the rig has, and none of that
        is worth solving on paper - so it is measured and corrected. The measured
        response of this page is close to median ~ light^0.5, hence the 1.8 exponent:
        a linear correction crept for six renders and never got there.
        """
        empty = self.render()
        for i in range(iters + 1):
            paper = empty[self._canvas_rect()]
            med = float(np.median(paper))
            clip = float((paper.min(axis=1) >= 254).mean())
            if (band[0] <= med <= band[1] and clip < 0.05) or i == iters:
                break
            self._scale_lights(float(np.clip((want / max(med, 1.0)) ** 1.8, 0.5, 2.0)))
            empty = self.render()
        return int(med), clip, empty

    def verify_shadow(self):
        """Prove RND_SHADOW is the shadow, by behaviour rather than by name.

        Shadows are a rendering flag on the scene, and the flag with the same index in
        the visualization option is a debug decoration: get the two arrays mixed and
        every call reports success while the rig films either a shadowless page or a
        grey ball hanging off the carriage. A real shadow darkens paper under the metal,
        and the pattern moves with the metal, so those three things are measured: two
        carriage positions, shadows on against shadows off.
        """
        self.ink.clear()
        self.upload()
        self.apply_cam(camera("hero", 0.5, self, focus=(0.0, 0.0)))
        page = float(self._page_polygon().sum())
        masks, areas, depths = [], [], []
        for y in (-0.45, 0.45):
            self.set_carriage((0.0, y), robot.LIFT * 0.5)
            self.set_shadow(False)
            off = self.render().astype(np.float32).mean(axis=2)
            self.set_shadow(True)
            on = self.render().astype(np.float32).mean(axis=2)
            band = ((on - off) < -12.0) & self._page_polygon()
            masks.append(band)
            areas.append(int(band.sum()) / max(page, 1.0))
            depths.append(float((off - on)[band].mean()) if band.any() else 0.0)
        self.park()
        self.set_shadow(False)
        both = (masks[0] & masks[1]).sum()
        iou = float(both) / max((masks[0] | masks[1]).sum(), 1.0)
        return min(areas), min(depths), iou

    def test_mark(self, xy):
        self.apply_cam(self._top_cam(self.top_orient))
        self.ink.clear()
        self.ink.splat(self.ink.cell(xy), MARK_R, MARK_RGB)
        self.upload()
        return self.render()

    def verify(self):
        worst, ns = 0.0, []
        n = self.ink.n
        for xy in VERIFY_POINTS:
            rc = self.ink.cell(xy)
            frame = self.test_mark(xy)
            m = self.find_mark(frame)
            assert m is not None, ("verification marker at (%.2f, %.2f) never rendered; it "
                                   "maps to texel (%.0f, %.0f) of a %d^2 page; %s"
                                   % (xy[0], xy[1], rc[0], rc[1], n, self.mark_stats(frame)))
            want = self.pixel_of(xy)
            worst = max(worst, float(np.linalg.norm(np.array(m[:2]) - want)))
            ns.append(m[2])
        # every marker is the same disc, so the same number of pixels: a marker that
        # came out small was eaten by the machine and its centroid is a half-disc's
        assert min(ns) > 0.6 * max(ns), ("a verification marker was occluded (%s px)" % ns)
        self.ink.clear()
        self.upload()
        return worst

    # -- whole-drawing check -------------------------------------------------
    def ink_centroid_check(self, out):
        """Where the ink actually is on screen vs where the pen says it is.

        A single marker only proves the map is right at one point. This compares the
        centroid of every dark pixel of the finished page with the centroid of the
        pen's own measured positions, which fails if the map is mirrored, rotated or
        simply wrong away from the calibration samples. Shadows are switched off for
        the measurement: the carriage's own shadow is a dark grey smudge near the
        corner it is parked in, and letting it into a centroid test would only prove
        that shadows exist.
        """
        pts = out["xy"][out["draw"]]
        pred = self.pixel_of(pts.mean(axis=0))
        self.set_shadow(False)
        try:
            got = self.find_ink(self.render())
        finally:
            self.set_shadow(True)
        assert got is not None, "no ink found in the verification render"
        return float(np.linalg.norm(np.array(got[:2]) - pred)), got[2]

    def _page_polygon(self):
        """The four projected sheet corners, filled: the page and nothing else."""
        pts = np.array([self.pixel_of((x, y)) for x in (-1.0, 1.0) for y in (-1.0, 1.0)])
        c = pts.mean(axis=0)
        q = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        img = Image.new("L", (self.width, self.height), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in q], fill=255)
        return np.asarray(img) > 0

    def shadow_depth(self, xy, kinds=("hero", "side")):
        """How much light the gantry takes off the paper, per pixel, in an angled shot.

        The one artefact every other check here is blind to, because it is lighting and
        not registration: a metre of metal over a white sheet casts a grey stripe across
        the drawing, and a film of a cat with a stripe through it looks broken. Measured
        by rendering the same frame with shadows on and off and comparing pixel by pixel,
        so the metal cancels and only the light it blocks is left. Only paper counts -
        ink, the machine and the background are not the thing the viewer is reading.
        """
        self.set_carriage(xy, robot.LIFT * 0.5)
        self.upload()
        worst, npaper = 0.0, 0
        for kind in kinds:
            self.apply_cam(camera(kind, 0.5, self, focus=tuple(xy)))
            lit = self._paper(shadow=False)
            dark = self._paper(shadow=True)
            m = np.isfinite(lit) & np.isfinite(dark)
            hi = float(np.percentile(lit[m], 95)) if m.any() else 0.0
            sel = m & (lit > 0.55 * hi) & (lit < 1.02 * hi)
            if not sel.any():
                continue
            r = np.clip(1.0 - dark[sel] / np.maximum(lit[sel], 1e-6), 0.0, None)
            worst = max(worst, float(np.percentile(r, 95)))
            npaper = int(sel.sum())
        self.set_shadow(False)
        return worst, npaper

    def _paper(self, shadow):
        """Mean luminance per pixel of the current page polygon, NaN off the sheet."""
        self.set_shadow(shadow)
        img = self.render().mean(axis=2)
        m = self._page_polygon()
        out = np.full(img.shape, np.nan)
        out[m] = img[m]
        return out


# ------------------------------------------------------------------- pixel work
def _box_axis(a, r, axis):
    n = a.shape[axis]
    idx = np.arange(n)
    lo = np.clip(idx - r, 0, n - 1)
    hi = np.clip(idx + r, 0, n - 1) + 1
    shape = [(s + (1 if i == axis else 0)) for i, s in enumerate(a.shape)]
    c = np.concatenate([np.zeros(shape, a.dtype), np.cumsum(a, axis=axis)], axis=axis)
    tot = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
    cnt = (hi - lo).reshape([(n if i == axis else 1) for i in range(a.ndim)])
    return tot / cnt


def blur(a, radius, passes=3):
    """Separable box blur, N passes ~ Gaussian. numpy only: no scipy, no cv2."""
    out = a.astype(np.float32)
    for _ in range(passes):
        out = _box_axis(out, radius, 0)
        out = _box_axis(out, radius, 1)
    return out


def tone(a, contrast=1.06, gamma=0.97, lift=0.0):
    a = np.clip((a - 0.5) * contrast + 0.5 + lift, 0.0, 1.0) ** gamma
    return a


_VIGNETTE = {}


def vignette(a, strength=0.28, softness=1.4):
    h, w = a.shape[:2]
    # the mask is a function of the frame size alone, and a film is one frame size for
    # its whole running time: rebuilding two million hypots every frame is the single
    # most expensive thing the grade does that has no reason to change between frames.
    key = (h, w, softness)
    m = _VIGNETTE.get(key)
    if m is None:
        y = (np.linspace(-1, 1, h)[:, None] * (h / float(max(h, w)))).astype(np.float32)
        x = (np.linspace(-1, 1, w)[None, :]).astype(np.float32)
        d = np.hypot(x, y) / np.sqrt(1.0 + (h / float(max(h, w))) ** 2)
        m = (1.0 - strength * np.clip(d, 0, 1) ** (2.0 / softness))[..., None]
        _VIGNETTE[key] = m
    return a * m


def grain(a, sigma=0.011, seed=0):
    rng = np.random.default_rng(seed)
    return a + rng.normal(0.0, sigma, a.shape[:2] + (1,)).astype(np.float32)


def unsharp(a, amount=0.4, radius=2):
    return np.clip(a + amount * (a - blur(a, radius)), 0.0, 1.0)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def overlay(a, lines, place="tl", color=(233, 240, 247), shadow=(5, 7, 12), margin=0.042):
    """Text drawn in image space. Labels and readouts only - never the artwork."""
    img = Image.fromarray(np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8))
    d = ImageDraw.Draw(img)
    h, w = a.shape[:2]
    fonts = [font(px) for _, px in lines]
    boxes = [d.textbbox((0, 0), txt, font=f)
             for (txt, _), f in zip(lines, fonts)]
    sizes = [(b[2] - b[0], b[3] - b[1]) for b in boxes]
    block = sum(s[1] + 0.5 * px for s, (_, px) in zip(sizes, lines))
    if place[0] == "t":
        y = margin * h
    elif place[0] == "m":
        # a title card wants its words on the optical centre, not on the bottom edge:
        # the bottom is where a caption sits *under* a picture, and a card is all picture.
        y = 0.5 * (h - block)
    else:
        y = h - margin * h - block
    for (txt, px), f, (tw, th) in zip(lines, fonts, sizes):
        if place[1] == "l":
            x = margin * w
        elif place[1] == "r":
            x = w - margin * w - tw
        else:
            x = 0.5 * (w - tw)
        d.text((x + 2, y + 2), txt, font=f, fill=shadow)
        d.text((x, y), txt, font=f, fill=color)
        y += th + 0.5 * px
    return np.asarray(img, np.float32) / 255.0


def grade(img, seed=0, sharp=0.4, vig=0.28, noi=0.011,
          shadow_rgb=(0.030, 0.040, 0.068)):
    a = tone(img.astype(np.float32) / 255.0)
    # a void of pure black reads as a screenshot. lifting only the deepest shadows
    # into a cold charcoal reads as a room the machine is standing in.
    s = np.clip(1.0 - a.max(axis=2, keepdims=True) / 0.20, 0.0, 1.0)
    a = a + s * np.asarray(shadow_rgb, np.float32)
    if sharp:
        a = unsharp(a, sharp)
    if vig:
        a = vignette(a, vig)
    if noi:
        a = grain(a, noi, seed)
    return np.clip(a, 0.0, 1.0)


# ------------------------------------------------------------------------ film
def paint_upto(scene, out, upto, width):
    """Fast-forward the page to program frame `upto`, painting the contact trace."""
    xy, fo = out["xy"], out["force"]
    scene.ink.clear()
    for i in range(int(upto)):
        j = min(i + 1, len(xy) - 1)
        if fo[i] <= F_MIN and fo[j] <= F_MIN:
            continue
        scene.ink.line(xy[i], xy[j], width)


def stroke_registration(scene, out, prog, width=LINE_W, min_len=2.0 * LINE_W):
    """Paint each stroke on its own and find it in the image, stroke by stroke.

    A single marker proves the map at one point. This proves it at every place the
    drawing actually occupies, using the pen's achieved positions, and it is the test
    that catches a registration error which happens to cancel out in the aggregate.

    Strokes shorter than a dot of ink are counted and returned instead of asserted on:
    one contact frame is a point, `ink.line` has no segment to paint, and a policy's
    degenerate micro-stroke is a fact about the drawing it made - not about whether the
    camera map is right, which is what this function exists to prove.
    """
    sid = np.asarray(prog["stroke_id"])
    dr = np.asarray(prog["draw"], bool)
    xy, fo = out["xy"], out["force"]
    if not dr.any():
        raise AssertionError("the program laid no ink at all: there is nothing to register")
    scene.apply_cam(scene._top_cam(scene.top_orient))
    scene.set_shadow(False)
    worst, rows, skip_n, skip_len = 0.0, [], 0, 0.0
    for k in range(int(sid[dr].max()) + 1):
        idx = np.flatnonzero(dr & (sid == k))
        if not len(idx):
            continue
        L = float(np.linalg.norm(np.diff(xy[idx], axis=0), axis=1).sum()) if len(idx) > 1 else 0.0
        if L < min_len:
            skip_n += 1
            skip_len += L
            continue
        scene.ink.clear()
        for i in idx[:-1]:
            j = i + 1
            if fo[i] > F_MIN or fo[j] > F_MIN:
                scene.ink.line(xy[i], xy[j], width)
        scene.upload()
        got = scene.find_ink(scene.render())
        assert got is not None, (
            "stroke %d carries %.4f u of pen travel and lays no measurable ink" % (k, L))
        pred = scene.pixel_of(xy[idx].mean(axis=0))
        e = float(np.hypot(got[0] - pred[0], got[1] - pred[1]))
        worst = max(worst, e)
        rows.append((int(sid[idx[0]]), e))
    scene.ink.clear()
    scene.upload()
    scene.set_shadow(True)
    return worst, rows, (skip_n, skip_len)


def camera_visibility(scene, kinds=("top", "hero", "macro", "side"), focus=(0.0, 0.0)):
    """Every shot family must actually see the ink, measured in pixels.

    A camera can be perfectly aimed at the underside of the table and still produce a
    handsome dark frame, and no amount of top-down calibration would notice. So each
    camera is shown the same five painted markers and has to find them.
    """
    pts = [(-0.85, -0.85), (0.85, -0.85), (-0.85, 0.85), (0.85, 0.85), (0.0, 0.0)]
    scene.ink.clear()
    for xy in pts:
        scene.ink.splat(scene.ink.cell(xy), MARK_R, MARK_RGB)
    scene.upload()
    rows = []
    for kind in kinds:
        scene.apply_cam(camera(kind, 0.5, scene, focus))
        img = scene.render()
        n = int((mark_weight(img) > 0.0).sum())
        rows.append((kind, n))
        assert n > 40, ("camera %r sees %d ink pixels: it is not looking at the page"
                        % (kind, n))
    scene.ink.clear()
    scene.upload()
    return rows


def camera(kind, p, scene, focus=(0.0, 0.0)):
    """One camera per shot family, all anchored to the page's own up direction.

    MuJoCo's elevation is negative when the camera sits above the subject, so every
    shot here is written with a negative angle: a positive one films the underside of
    the table. And the flat lay is measured to put canvas +y at the top of the frame
    at azimuth `up`, so the angled shots start from that same azimuth and only lean
    away from it - a camera with an unrelated azimuth films the cat lying on its side.
    """
    s, sc = scene.scale, scene
    up = sc.top_orient[1]
    if kind == "top":
        return sc._top_cam(sc.top_orient, 1.24 * 1.15 * s / (1.0 + 0.14 * p))
    if kind == "hero":
        return {"orthographic": False, "elevation": -(52.0 + 8.0 * p),
                "azimuth": up - 16.0 + 10.0 * p,
                "distance": _dist(sc, 0.128 - 0.006 * p), "lookat": (0, 0, 0.004)}
    if kind == "macro":
        return {"orthographic": False, "elevation": -(38.0 + 10.0 * p),
                "azimuth": up + 22.0 - 12.0 * p, "distance": _dist(sc, 0.048),
                "lookat": (focus[0] * s, focus[1] * s, 0.004)}
    if kind == "side":
        return {"orthographic": False, "elevation": -(26.0 + 10.0 * p),
                "azimuth": up - 34.0 + 24.0 * p, "distance": _dist(sc, 0.085),
                "lookat": (focus[0] * s, focus[1] * s, 0.004)}
    raise ValueError("unknown camera %r" % kind)


def default_edl(T):
    """A five-shot edit that walks the whole program once, in program order.

    Shots never go backwards in program time: the page only ever gains ink, so an
    out-of-order edit would show a drawing that has not happened yet.

    And nothing films the page from straight above while the pen is down. Seen from
    the flat lay the bridge and the head are two black bars that walk across the
    drawing, and they are only ever *above* it, so they cover whatever they pass; the
    one top-down shot here is the reveal, whose carriage has been parked clear of the
    sheet first. The moving shots lean, which puts the machine to one side of the page
    and casts its shadow beside the ink instead of over it.
    """
    a, b = 0, T
    return [
        {"name": "setup", "t0": a, "t1": a + 90, "rate": 2.0, "cam": "hero",
         "hud": False, "still": None},
        {"name": "opening", "t0": a + 90, "t1": int(0.40 * b), "rate": 4.0, "cam": "side",
         "hud": True},
        {"name": "contact", "t0": int(0.40 * b), "t1": int(0.62 * b), "rate": 1.6,
         "cam": "macro", "hud": True},
        {"name": "sweep", "t0": int(0.62 * b), "t1": int(0.88 * b), "rate": 4.0,
         "cam": "hero", "hud": True},
        {"name": "finish", "t0": int(0.88 * b), "t1": b, "rate": 2.0, "cam": "macro",
         "hud": True},
        {"name": "reveal", "t0": b, "t1": b, "rate": 0.0, "cam": "top", "hud": False,
         "still": 110, "park": True},
    ]


class Film:
    """Turns a simulated program into a graded, captioned sequence of shots."""

    def __init__(self, scene, out, prog, width=LINE_W, fps=30.0, title=None, credits=None):
        self.scene, self.out, self.prog = scene, out, prog
        self.width, self.fps = float(width), float(fps)
        self.title = title or []
        self.credits = credits or []
        self.T = int(out["T"])
        self.names = list(prog.get("names", []))
        # what the viewer wants is the last stroke that actually laid ink, not the
        # stroke the head happens to be flying over on its way somewhere
        dr = np.asarray(prog["draw"], bool)
        self.last_stroke = np.maximum.accumulate(np.where(dr, np.asarray(prog["stroke_id"]), -1))

    def hud(self, i, force):
        sid = int(self.last_stroke[i]) if len(self.last_stroke) else -1
        n = max(1, len(self.names))
        err = float(self.out["err"][i])
        return [("stroke %2d / %d" % (max(0, sid) + 1, n), 22),
                ("contact %.2f N   err %.4f u" % (force, err), 20)]

    def frame(self, i, k, nk, shot, focus=None):
        sc = self.scene
        # a flat lay wants an evenly lit page; an angled shot wants the machine's shadow
        sc.set_shadow(shot["cam"] != "top")
        if shot.get("park"):
            # the reveal is of the drawing, not of the head that made it
            sc.park()
        else:
            # pen height, from the joint travel the simulation actually recorded
            sc.set_carriage(self.out["xy"][i], float(self.out["pen"][i]) * robot.LIFT)
        p = k / max(1, nk - 1)
        sc.apply_cam(camera(shot["cam"], p, sc, focus or tuple(self.out["xy"][i])))
        sc.upload()
        a = grade(sc.render(), seed=i)
        lines = []
        if shot.get("hud") and i < self.T:
            lines += self.hud(i, float(self.out["force"][i]))
        if shot.get("caption"):
            lines.append((shot["caption"], 26))
        if lines:
            a = overlay(a, lines, "tl")
        return a

    def shot(self, spec):
        sc = self.scene
        t0, t1 = int(spec["t0"]), min(int(spec["t1"]), self.T)
        paint_upto(sc, self.out, t0, self.width)
        sc.upload()
        if spec.get("still"):
            nk = int(spec["still"])
            i = max(0, self.T - 1)
            for k in range(nk):
                yield dict(i=i, k=k, nk=nk, shot=spec, p=k / max(1, nk - 1.0)), k
            return
        rate = max(1e-6, float(spec["rate"]))
        nk = max(1, int(np.ceil((t1 - t0) / rate)))
        xy, fo = self.out["xy"], self.out["force"]
        painted = t0
        for k in range(nk):
            i = min(self.T - 1, t0 + int(round(k * rate)))
            # every program frame between the last one shown and this one has to be on
            # the page now: a shot that samples one segment in four would otherwise film
            # the pen's line as a row of dots, and the ink it skipped never comes back.
            while painted < i:
                a, b = painted, min(painted + 1, self.T - 1)
                if fo[a] > F_MIN or fo[b] > F_MIN:
                    sc.ink.line(xy[a], xy[b], self.width)
                painted += 1
            j = min(self.T - 1, i + 1)
            if fo[i] > F_MIN or fo[j] > F_MIN:
                sc.ink.line(xy[i], xy[j], self.width)
            yield dict(i=i, k=k, nk=nk, shot=spec, p=k / max(1, nk - 1.0)), k

    def title_card(self, spec, a):
        if not self.title:
            return a
        return overlay(a, self.title, "tl")

    def build(self, path, edl=None, log=None, on_frame=None):
        import imageio
        edl = edl or default_edl(self.T)
        n_out = 0
        # x264 writes the index at close, so an interrupted render leaves a file that
        # exists, is huge, and cannot be played. Write under a .part name and move it
        # into place only once the writer closed cleanly, so no half film can ever be
        # mistaken for a finished one - and keep the .mp4 tail, because that is how
        # imageio picks its backend.
        tmp = path + ".part.mp4"
        writer = imageio.get_writer(tmp, fps=self.fps, codec="libx264",
                                    macro_block_size=1, pixelformat="yuv420p",
                                    quality=8)
        try:
            for spec in edl:
                for meta, k in self.shot(spec):
                    a = self.frame(meta["i"], k, meta["nk"], spec)
                    if spec["name"] == "setup":
                        a = self.title_card(spec, a)
                    if spec["name"] == "reveal" and self.credits:
                        a = overlay(a, self.credits, "tl")
                    writer.append_data((np.clip(a, 0, 1) * 255.0).astype(np.uint8))
                    n_out += 1
                    if on_frame:
                        on_frame(spec["name"], n_out)
        except BaseException:
            writer.close()
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        writer.close()
        os.replace(tmp, path)
        if log:
            log.write("film %s frames %d shots %d\n" % (path, n_out, len(edl)))
        return n_out


# ----------------------------------------------------------------------- main
def stills(scene, out, prog, path, width=LINE_W, at=0.55):
    """The four shot families at one instant of the program, graded, for judging by eye."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    T = int(out["T"])
    film = Film(scene, out, prog, width=width)
    i = min(T - 1, int(at * T))
    paint_upto(scene, out, i, width)
    tiles = []
    for kind in ("top", "hero", "macro", "side"):
        shot = {"name": kind, "cam": kind, "hud": True, "park": at >= 1.0}
        tiles.append(film.frame(i, 0, 1, shot))
    grid = np.concatenate([np.concatenate([tiles[0], tiles[1]], axis=1),
                           np.concatenate([tiles[2], tiles[3]], axis=1)], axis=0)
    Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


if __name__ == "__main__":
    import sys
    import art
    import planner

    draft = "--draft" in sys.argv
    want_film = "--film" in sys.argv
    res = (1280, 720) if draft else (1920, 1080)

    hero = art.fit_canvas(art.cat_strokes())
    pr = planner.plan(hero, first="head")
    rob = robot.Plotter()
    out = rob.run(pr)

    scene = Scene(width=res[0], height=res[1])
    assert scene.calib_err_px < REG_GATE, "ink would land on the wrong part of the page"

    worst, rows, (skip_n, skip_len) = stroke_registration(scene, out, pr)
    assert rows, "no stroke is long enough to lay a measurable dot of ink"
    assert worst < REG_GATE, "a stroke landed %.1f px from where the pen drew it" % worst
    print("REGISTRATION  %d strokes, worst %.2f px, median %.2f px (gate %.1f px)"
          % (len(rows), worst, float(np.median([e for _, e in rows])), REG_GATE))
    print("              %d strokes too short to register (%.4f u of pen travel)"
          % (skip_n, skip_len))

    vis = camera_visibility(scene)
    print("CAMERAS   " + "  ".join("%s %d ink px" % (k, n) for k, n in vis))

    depth, npaper = scene.shadow_depth(out["xy"][int(0.55 * out["T"])])
    # a shadow is wanted - metal over paper that casts nothing is a sticker - but not a
    # second drawing laid over the first. measured on the last 5% of the lit paper, so
    # one hard-edged stripe cannot hide behind an average.
    assert depth < SHADOW_GATE, (
        "the gantry shadow takes %.0f%% of the light off the worst-lit 5%% of the "
        "sheet: the film shows a grey stripe across the drawing" % (100.0 * depth))
    print("SHADOW    worst 5%% of the paper loses %.3f of its light (%d px, hero and side)"
          "  (gate %.2f, depth map %d)" % (depth, npaper, SHADOW_GATE, SHADOW_SIZE))

    paint_upto(scene, out, out["T"], LINE_W)
    scene.park()
    scene.apply_cam(scene._top_cam(scene.top_orient))
    scene.upload()
    dpx, npix = scene.ink_centroid_check(out)
    assert dpx < CENTROID_GATE, "the finished page is %.1f px off where the pen says it is" % dpx
    covered = scene.ink.area()
    target = float(planner.stats(pr)["ink_length"]) * LINE_W
    assert 0.55 * target < covered < 1.9 * target, (
        "page holds %.4f u^2 of ink, the program drew %.4f u^2" % (covered, target))
    print("RENDER PASS  whole-page centroid %.1f px over %.0f ink px  area %.4f u^2"
          " (program %.4f u^2, gate %.1f px)" % (dpx, npix, covered, target, CENTROID_GATE))

    tag = "_draft" if draft else ""
    for at, what in ((0.55, "mid"), (1.0, "done")):
        p = stills(scene, out, pr, os.path.join(
            OUT, "render_stills_%s%s.png" % (what, tag)), at=at)
        print("stills %s" % p)

    if want_film:
        os.makedirs(OUT, exist_ok=True)
        mp4 = os.path.join(OUT, "cat_film%s.mp4" % ("_draft" if draft else ""))
        import time
        t0 = time.time()
        credits = [("MuJoCo %s   pen %.1f mm   contact %.2f N"
                    % (mujoco.__version__, 2 * robot.TIP_R * 1000,
                       float(out["force"][out["draw"]].mean())), 20),
                   ("ink tracking rms %.5f canvas units" %
                    float(np.sqrt((out["err"][out["draw"]] ** 2).mean())), 20)]
        film = Film(scene, out, pr, title=[("DIFFUSION  POLICY  CAT", 46),
                                           ("a pen, a page, and a machine that obeys", 24)],
                    credits=credits)
        n = film.build(mp4, on_frame=lambda s, k: None if k % 100 else print("  %s %d" % (s, k)))
        print("FILM %s  %d frames  %.0f s at %.0f fps  rendered in %.1f s"
              % (mp4, n, n / film.fps, film.fps, time.time() - t0))
