#!/usr/bin/env bash

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

# This script updates `staging/src/k8s.io/kubectl/pkg/util/i18n/translations/kubectl/template.pot` and
# generates/fixes .po and .mo files.
# Usage: `update-translations.sh`.

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/util.sh"

TRANSLATIONS="staging/src/k8s.io/kubectl/pkg/util/i18n/translations"
KUBECTL_FILES=()
KUBECTL_DEFAULT_LOCATIONS=(
  "pkg/kubectl/cmd"
  "staging/src/k8s.io/kubectl/pkg/cmd"
)
KUBECTL_IGNORE_FILES_REGEX="cmd/kustomize/kustomize.go"

generate_pot="false"
generate_mo="false"
fix_translations="false"

while getopts "hf:xkg" opt; do
  case ${opt} in
    h)
      echo "$0 [-f files] [-x] [-k] [-g]"
      echo " -f <file-path>: Files to process"
      echo " -x extract strings to a POT file"
      echo " -k fix .po files; deprecate translations by marking them obsolete and supply default messages"
      echo " -g sort .po files and generate .mo files"
      exit 0
      ;;
    f)
      KUBECTL_FILES+=("${OPTARG}")
      ;;
    x)
      generate_pot="true"
      ;;
    k)
      fix_translations="true"
      ;;
    g)
      generate_mo="true"
      ;;
    \?)
      echo "[-f <files>] -x -g" >&2
      exit 1
      ;;
  esac
done

if [[ ${#KUBECTL_FILES} -eq 0 ]]; then
  KUBECTL_FILES+=("${KUBECTL_DEFAULT_LOCATIONS[@]}")
fi

if ! which go-xgettext > /dev/null; then
  echo 'Can not find go-xgettext, install with:'
  echo 'go get github.com/gosexy/gettext/go-xgettext'
  exit 1
fi

if ! which msgfmt > /dev/null; then
  echo 'Can not find msgfmt, install with:'
  echo 'apt-get install gettext'
  exit 1
fi

if [[ "${generate_pot}" == "true" ]]; then
  echo "Extracting strings to POT"
  # shellcheck disable=SC2046
  go-xgettext -k=i18n.T $(grep -lr "i18n.T" "${KUBECTL_FILES[@]}" | grep -vE "${KUBECTL_IGNORE_FILES_REGEX}") > tmp.pot
  perl -pi -e 's/CHARSET/UTF-8/' tmp.pot
  perl -pi -e 's/\\(?!n"\n|")/\\\\/g' tmp.pot
  kube::util::ensure-temp-dir
  if msgcat -s tmp.pot > "${KUBE_TEMP}/template.pot"; then
    mv "${KUBE_TEMP}/template.pot" "${TRANSLATIONS}/kubectl/template.pot"
    rm tmp.pot
  else
    echo "Failed to update template.pot"
    exit 1
  fi
fi

if [[ "${fix_translations}" == "true" ]]; then
  echo "Fixing .po files"
  kube::util::ensure-temp-dir
  for PO_FILE in "${TRANSLATIONS}"/kubectl/*/LC_MESSAGES/k8s.po; do
    TMP="${KUBE_TEMP}/fix.po"
    if [[ "${PO_FILE}" =~ .*/default/.* || "${PO_FILE}" =~ .*/en_US/.*  ]]; then
      # mark obsolete, and set default values for english translations
      msgen "${TRANSLATIONS}/kubectl/template.pot" | \
        msgmerge -s --no-fuzzy-matching "${PO_FILE}" - > "${TMP}"
    else
      # mark obsolete, but do not add untranslated messages
      msgmerge -s --no-fuzzy-matching "${PO_FILE}" "${TRANSLATIONS}/kubectl/template.pot" | msgattrib --translated - > "${TMP}"
    fi
    mv "${TMP}" "${PO_FILE}"
  done
fi

if [[ "${generate_mo}" == "true" ]]; then
  echo "Generating .po and .mo files"
  for x in "${TRANSLATIONS}"/*/*/*/*.po; do
    msgcat -s "${x}" > tmp.po
    mv tmp.po "${x}"
    echo "generating .mo file for: ${x}"
    msgfmt "${x}" -o "$(dirname "${x}")/$(basename "${x}" .po).mo"
  done
fi
