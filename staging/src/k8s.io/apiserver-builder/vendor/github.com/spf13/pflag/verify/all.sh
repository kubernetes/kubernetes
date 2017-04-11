#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

ROOT=$(dirname "${BASH_SOURCE}")/..

# Some useful colors.
if [[ -z "${color_start-}" ]]; then
  declare -r color_start="\033["
  declare -r color_red="${color_start}0;31m"
  declare -r color_yellow="${color_start}0;33m"
  declare -r color_green="${color_start}0;32m"
  declare -r color_norm="${color_start}0m"
fi

SILENT=true

function is-excluded {
  for e in $EXCLUDE; do
    if [[ $1 -ef ${BASH_SOURCE} ]]; then
      return
    fi
    if [[ $1 -ef "$ROOT/hack/$e" ]]; then
      return
    fi
  done
  return 1
}

while getopts ":v" opt; do
  case $opt in
    v)
      SILENT=false
      ;;
    \?)
      echo "Invalid flag: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if $SILENT ; then
  echo "Running in the silent mode, run with -v if you want to see script logs."
fi

EXCLUDE="all.sh"

ret=0
for t in `ls $ROOT/verify/*.sh`
do
  if is-excluded $t ; then
    echo "Skipping $t"
    continue
  fi
  if $SILENT ; then
    echo -e "Verifying $t"
    if bash "$t" &> /dev/null; then
      echo -e "${color_green}SUCCESS${color_norm}"
    else
      echo -e "${color_red}FAILED${color_norm}"
      ret=1
    fi
  else
    bash "$t" || ret=1
  fi
done
exit $ret
