#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# Script to generate docs from the latest swagger spec.

set -o errexit
set -o nounset
set -o pipefail

cd /build

# gendocs takes "input.json" as the input swagger spec.
# $1 is expected to be <group>_<version>
cp /swagger-source/"$1".json input.json

./gradle-2.5/bin/gradle gendocs --info

#insert a TOC for top level API objects
buf="== Top Level API Objects\n\n"
top_level_models=$(grep '&[A-Za-z]*{},' /register.go | sed 's/.*&//;s/{},//')

# check if the top level models exist in the definitions.adoc. If they exist,
# their name will be <version>.<model_name>
VERSION="${1#*_}"
for m in $top_level_models
do
  if grep -xq "=== ${VERSION}.$m" ./definitions.adoc
  then
    buf+="* <<${VERSION}.$m>>\n"
  fi
done
sed -i "1i $buf" ./definitions.adoc

# fix the links in .adoc, replace <<x.y>> with link:definitions.html#_x_y[x.y], and lowercase the _x_y part
sed -i -e 's|<<\(.*\)\.\(.*\)>>|link:#_\L\1_\2\E[\1.\2]|g' ./definitions.adoc
sed -i -e 's|<<\(.*\)\.\(.*\)>>|link:../definitions#_\L\1_\2\E[\1.\2]|g' ./paths.adoc

# fix the link to <<any>>
sed -i -e 's|<<any>>|link:#_any[any]|g' ./definitions.adoc
sed -i -e 's|<<any>>|link:../definitions#_any[any]|g' ./paths.adoc

# change the title of paths.adoc from "paths" to "operations"
sed -i 's|== Paths|== Operations|g' ./paths.adoc

# $$ has special meaning in asciidoc, we need to escape it
sed -i 's|\$\$|+++$$+++|g' ./definitions.adoc

echo -e "=== any\nRepresents an untyped JSON map - see the description of the field for more info about the structure of this object." >> ./definitions.adoc

asciidoctor definitions.adoc
asciidoctor paths.adoc

cp definitions.html /output/
cp paths.html /output/operations.html

echo "SUCCESS"
