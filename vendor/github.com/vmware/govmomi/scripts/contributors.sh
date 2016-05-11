#!/bin/bash

set -e

outfile="CONTRIBUTORS"
tmpfile="CONTRIBUTORS.tmp"
cp "${outfile}" "${tmpfile}"

# Make sure the email address of every contributor is listed
git shortlog -sne | while read line; do
  name=$(perl -pe 's/\d+\s+//' <<<"${line}")
  email=$(grep -Po '(?<=<).*(?=>)' <<<"${name}")
  if ! grep -q "${email}" "${outfile}"; then
    echo "${name}" >> "${tmpfile}"
  fi
done

# Sort entries
(
  sed -ne '1,5p' "${tmpfile}"
  sed -ne '1,5!p' "${tmpfile}" | sort
) > "${outfile}"

rm -f "${tmpfile}"
