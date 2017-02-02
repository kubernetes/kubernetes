#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

KUBECTL_FILES="pkg/kubectl/cmd/*.go pkg/kubectl/cmd/*/*.go"

if ! which go-xgettext > /dev/null; then
  echo 'Can not find go-xgettext, install with:'
  echo 'go get github.com/gosexy/gettext/go-xgettext'
  exit 1
fi

go-xgettext -k=i18n.T ${KUBECTL_FILES} > tmp.pot
msgcat -s tmp.pot > translations/kubectl/template.pot
rm tmp.pot

for x in translations/*/*/*/*.po; do
  msgcat -s $x > tmp.po
  mv tmp.po $x
  echo "generating .mo file for: $x"
  msgfmt $x -o "$(dirname $x)/$(basename $x .po).mo"
done

./hack/generate-bindata.sh
