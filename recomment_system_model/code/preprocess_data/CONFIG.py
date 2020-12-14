import os
from pathlib import Path



link = Path(os.path.abspath(__file__))
LINK = link.parent.parent.parent
LINK_DATA = os.path.join(LINK,"data")
print(LINK_DATA)