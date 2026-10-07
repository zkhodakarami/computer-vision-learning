"""
One-time fix for a certificate error on Mac.

Run it with the project's Python, from the project folder:
    python 00_setup/fix_certificates.py

Why this exists:
    The Python you download from python.org on a Mac comes without the list of
    trusted website certificates. So when a lesson tries to download a dataset,
    Python refuses with "CERTIFICATE_VERIFY_FAILED".

What this does:
    Writes a tiny file into the .venv folder that tells Python to use the trusted
    list that comes with the "certifi" library. It runs automatically every time
    the project's Python starts, so you only do this once.
"""

import site
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    print("Please turn on the project's environment first:  source .venv/bin/activate")
    sys.exit(1)

site_packages = Path(site.getsitepackages()[0])
target = site_packages / "sitecustomize.py"
target.write_text(
    '"""Written by 00_setup/fix_certificates.py: lets Python trust normal websites."""\n'
    "import os\n"
    "try:\n"
    "    import certifi\n"
    "    os.environ.setdefault('SSL_CERT_FILE', certifi.where())\n"
    "except ImportError:\n"
    "    pass\n"
)
print("Wrote", target)
print("Done. Dataset downloads should work now.")
