#!/bin/bash

DEST=$2
PACKAGE=$3
TMPFILE="mockgen_tmp.go"
# uppercase the name of the interface
ORIG_INTERFACE_NAME=$4
INTERFACE_NAME="$(tr '[:lower:]' '[:upper:]' <<< ${ORIG_INTERFACE_NAME:0:1})${ORIG_INTERFACE_NAME:1}"

# Gather all files that contain interface definitions.
# These interfaces might be used as embedded interfaces,
# so we need to pass them to mockgen as aux_files.
AUX=()
for f in *.go; do
  if [[ -z ${f##*_test.go} ]]; then
    # skip test files
    continue;
  fi
  if $(egrep -qe "type (.*) interface" $f); then
    AUX+=("github.com/lucas-clemente/quic-go=$f")
  fi
done

# Find the file that defines the interface we're mocking.
for f in *.go; do
  if [[ -z ${f##*_test.go} ]]; then
    # skip test files
    continue;
  fi
  INTERFACE=$(sed -n "/^type $ORIG_INTERFACE_NAME interface/,/^}/p" $f)
  if [[ -n "$INTERFACE" ]]; then
    SRC=$f
    break
  fi
done

if [[ -z "$INTERFACE" ]]; then
  echo "Interface $ORIG_INTERFACE_NAME not found."
  exit 1
fi

AUX_FILES=$(IFS=, ; echo "${AUX[*]}")

## create a public alias for the interface, so that mockgen can process it
echo -e "package $1\n" > $TMPFILE
echo "$INTERFACE" | sed "s/$ORIG_INTERFACE_NAME/$INTERFACE_NAME/" >> $TMPFILE
goimports -w $TMPFILE
mockgen -package $1 -self_package $3 -destination $DEST -source=$TMPFILE -aux_files $AUX_FILES
goimports -w $DEST
sed "s/$TMPFILE/$SRC/" "$DEST" > "$DEST.new" && mv "$DEST.new" "$DEST"
rm "$TMPFILE"
