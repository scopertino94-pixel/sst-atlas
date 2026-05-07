import sys
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def run_oisst(req):
    print(f"[DEBUG] run_oisst called: mode={req.mode} date={req.month}/{req.day}/{req.year} region={req.region}")

    try:
        import oisst_main
    except Exception as e:
        import traceback
        print(f"[DEBUG] IMPORT FAILED: {e}")
        traceback.print_exc()
        raise e

    import importlib
    print("[DEBUG] reloading oisst_main...")
    target_date = f"{req.month:02d}/{req.day:02d}/{req.year}"
    oisst_main.MODE                 = req.mode
    oisst_main.TARGET_DATE          = target_date
    oisst_main.BASELINE_DATE        = getattr(req, 'baseline_date', '4/18/2015')
    oisst_main.REGION               = req.region
    oisst_main.THEME                = req.theme
    oisst_main.REMOVE_GLOBAL_MEAN   = req.remove_global_mean
    oisst_main.SHOW_OCEANIC_INDICES = req.show_oceanic_indices
    oisst_main.SHOW_PCT_OVERLAY     = req.show_pct_overlay
    oisst_main.SHOW_INSET_MAP       = req.show_inset_map

    print(f"[DEBUG] config set. TARGET_DATE={oisst_main.TARGET_DATE}, calling main()...")

    buf = io.BytesIO()
    _original_savefig = plt.savefig
    _original_show = plt.show

    def patched_savefig(fname, *args, **kwargs):
        print(f"[DEBUG] patched_savefig called!")
        kwargs.pop('bbox_inches', None)
        _original_savefig(buf, *args, format='png', bbox_inches='tight', **kwargs)

    def patched_show(*args, **kwargs):
        print(f"[DEBUG] patched_show called!")
        pass

    plt.savefig = patched_savefig
    plt.show = patched_show

    try:
        oisst_main.main()
        print("[DEBUG] main() completed!")
    except SystemExit:
        print("[DEBUG] SystemExit caught")
        pass
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception: {e}")
        traceback.print_exc()
        raise e
    finally:
        plt.savefig = _original_savefig
        plt.show = _original_show
        plt.close('all')

    buf.seek(0)
    data = buf.read()
    print(f"[DEBUG] buf size: {len(data)} bytes")
    if not data:
        raise RuntimeError("Script ran but produced no image — plt.savefig may not have been called.")
    return data

def run_ersst(req):
    import ersst_v6
    import importlib

    importlib.reload(ersst_v6)

    buf = io.BytesIO()
    _original_savefig = plt.savefig
    _original_show = plt.show

    def patched_savefig(fname, *args, **kwargs):
        kwargs.pop('bbox_inches', None)
        _original_savefig(buf, *args, format='png', bbox_inches='tight', **kwargs)

    def patched_show(*args, **kwargs):
        pass

    plt.savefig = patched_savefig
    plt.show = patched_show

    try:
        # ersst_v6 has no main() — reload executes the script directly
        importlib.reload(ersst_v6)
    except SystemExit:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    finally:
        plt.savefig = _original_savefig
        plt.show = _original_show
        plt.close('all')

    buf.seek(0)
    data = buf.read()
    if not data:
        raise RuntimeError("Script ran but produced no image — plt.savefig may not have been called.")
    return data