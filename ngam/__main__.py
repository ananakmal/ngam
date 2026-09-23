"""
ngam.__main__
=============
Direct module execution entrypoint (`python -m ngam`).
"""

import sys
from ngam.cli import main

if __name__ == "__main__":
    sys.exit(main())
