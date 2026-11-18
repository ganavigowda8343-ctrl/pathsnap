#!/bin/sh
# Rebuild report/PathSnap_Report.pdf from report/report.html (uses Google Chrome headless).
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --allow-file-access-from-files --no-pdf-header-footer \
  --virtual-time-budget=10000 --print-to-pdf="$(pwd)/PathSnap_Report.pdf" "file://$(pwd)/report.html" 2>/dev/null
echo "Wrote $(pwd)/PathSnap_Report.pdf"
