#!/bin/bash -eu
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To regenerate the CEL lexer/parser statically do the following:
# 1. Download the latest anltr tool from https://www.antlr.org/download.html
# 2. Copy the downloaded jar to the gen directory. It will have a name
#    like antlr-<version>-complete.jar.
# 3. Modify the script below to refer to the current ANTLR version.
# 4. Execute the generation script from the gen directory.
# 5. Delete the jar and commit the regenerated sources.

#!/bin/sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Generate AntLR artifacts.
java -Xmx500M -cp ${DIR}/antlr-4.12.0-complete.jar org.antlr.v4.Tool  \
    -Dlanguage=Go \
    -package gen \
    -o ${DIR} \
    -visitor ${DIR}/CEL.g4

