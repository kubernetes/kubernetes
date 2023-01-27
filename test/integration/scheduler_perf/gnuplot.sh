#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Invoke this script with a list of *.dat and it'll plot them with gnuplot.
# Any non-file parameter is passed through to gnuplot. By default,
# an X11 window is used to display the result. To write into a file,
# use
#  -e "set term png; set output <output>.png"

files=()
args=( -e "set term x11 persist" )

for i in "$@"; do
    if [ -f "$i" ]; then
        files+=("$i")
    else
        args+=("$i")
    fi
done

(
    cat <<EOF
set ytics autofreq nomirror tc lt 1
set ylabel 'scheduled' tc lt 1
set y2tics autofreq nomirror tc lt 2
set y2label 'attempts/scheduled' tc lt 2
EOF
    echo -n "plot "
    for file in "${files[@]}"; do
        echo -n "'${file}' using 1:2 with linespoints title '$(basename "$file" .dat) scheduled' axis x1y1, "
        echo -n "'${file}' using 1:(\$3/\$2) with linespoints title '$(basename "$file" .dat) attempts/scheduled' axis x1y2, "
    done
    echo
) | tee /dev/stderr | gnuplot "${args[@]}" -
