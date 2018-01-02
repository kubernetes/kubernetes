#!/usr/bin/env bash

#
# This script generates a gitdm compatible email aliases file from a git
# formatted .mailmap file.
#
# Usage:
#  $> ./generate_aliases <mailmap_file> > aliases
#

cat $1 | \
    grep -v '^#' | \
    sed 's/^[^<]*<\([^>]*\)>/\1/' | \
    grep '<.*>' | sed -e 's/[<>]/ /g' | \
    awk '{if ($3 != "") { print $3" "$1 } else {print $2" "$1}}' | \
    sort | uniq
