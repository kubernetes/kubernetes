#!/bin/bash -e

# pipe the most recent debug run to xmlformat
cd ${GOVC_DEBUG_PATH-"$HOME/.govmomi/debug"}
cd $(ls -t | head -1)

header() {
    printf "<!-- %s %s/%s\n%s\n-->\n" "$1" "$PWD" "$2" "$(tr -d '\r' < "$3")"
}

for file in *.req.xml; do
    base=$(basename "$file" .req.xml)
    header Request "$file" "${base}.req.headers"
    xmlformat < "$file"
    file="${base}.res.xml"
    header Response "$file" "${base}.res.headers"
    xmlformat < "$file"
done
