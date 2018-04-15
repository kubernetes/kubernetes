#!/usr/bin/env bash
# Copyright 2016 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

if ! which -s bazel && [[ "${CALLED_FROM_MAIN_MAKEFILE:-}" == "" ]]; then
  echo "Please use 'make update' or install https://bazel.build"
fi

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::util::ensure-gnu-sed

# Remove generated files prior to running kazel.
# TODO(spxtr): Remove this line once Bazel is the only way to build.
rm -f "${KUBE_ROOT}/pkg/generated/openapi/zz_generated.openapi.go"

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install tools we need, but only from vendor/...
go install ./vendor/github.com/bazelbuild/bazel-gazelle/cmd/gazelle

go install ./vendor/github.com/kubernetes/repo-infra/kazel

cd "${KUBE_ROOT}"
go get github.com/bazelbuild/buildtools/buildozer
touch "${KUBE_ROOT}/vendor/BUILD"

dozer() {
  # returns 0 on successful mod and 3 on no change
  buildozer --quiet "$@" && return 0 || [[ "$?" == 3 ]] && return 0
  return 1
}

gazelle-fix() {
  find . -name BUILD -or -name BUILD.bazel | xargs sed -i '' -e 's|//build:go.bzl|@io_bazel_rules_go//go:def.bzl|g'
  gazelle fix \
      -build_file_name=BUILD,BUILD.bazel \
      -external=vendored \
      -proto=legacy \
      -mode=fix
  # gazelle gets confused by our staging/ directory, prepending an extra
  # "k8s.io/kubernetes/staging/src" to the import path.
  # gazelle won't follow the symlinks in vendor/, so we can't just exclude
  # staging/. Instead we just fix the bad paths with sed.
  find staging -name BUILD -o -name BUILD.bazel | \
    xargs ${SED} -i 's|\(importpath = "\)k8s.io/kubernetes/staging/src/\(.*\)|\1\2|'
}

gazelle-fix

kazel

ensure-rule() {
  if ! bazel query "$1:all" | grep "$1:$3" ; then
    dozer "new $2 $3" "$1:__pkg__"
  else
    dozer "set kind $2" "$1:$3"
  fi
}

update-k8s-gengo() {
  local name="$1" # something like deepcopy
  local pkg="$2"  # something like k8s.io/code-generator/cmd/deepcopy-gen
  local out="$3"  # something like zz_generated.deepcopy.go
  local match="$4"  # something like +k8s:deepcopy-gen=
  local extra=("${@:5}")  # list of extra packages

  local all_rule="k8s_${name}_all"  # k8s_deepcopy_all, which generates out for matching packages
  local rule="k8s_${name}" # k8s_deepcopy, which copies out the file for a particular package

  # look for packages that contain match
  echo "Looking for packages with a $match comment in go code..."

  white=($(find . -name *.go | \
      grep -v "$pkg" | \
      (xargs grep -l "$match" || true)))
  want=()
  for w in "${white[@]}"; do
    if grep "$match" "$w" | grep -q -v "${match}false"; then
      want+=( "$(dirname "$w" | $SED -e 's|./staging/src/|./vendor/|;s|^./||')" )
    else
      echo SKIP: $w, has only ${match}false tags
    fi
  done
  want=($(IFS=$'\n ' ; echo "${want[*]}" | sort -u))
  deps=()
  for w in "${want[@]}"; do
    deps+=( "./$w" )
  done
  deps=( $(go list -f "{{.ImportPath}}{{\"\n\"}}{{range .Deps}}{{.}}{{\"\n\"}}{{end}}" "${deps[@]}" ${extra:+"${extra[@]}"} | sort -u | grep -E '^k8s.io/kubernetes/' | $SED -e 's|^k8s.io/kubernetes/||') )

  # Ensure that k8s_deepcopy_all() rule exists
  dozer "new_load //build:deepcopy.bzl $all_rule" //build:__pkg__
  ensure-rule //build "$all_rule" "${name}-sources"
  dozer "set packages ${want[*]}" "//build:$name-sources"
  dozer "set deps ${deps[*]}" "//build:$name-sources"
  dozer "add srcs :$name-sources" //build:all-generated-sources
  have=$(find . -name BUILD -or -name BUILD.bazel | (xargs grep -l "$rule(" || true))
  if [[ -n "$have" ]]; then
    echo Deleting existing "$rule" commands...
    $SED -i -e "/^$rule/d" $have
  fi

  echo "Adding $rule() rules"
  for w in "${want[@]}"; do
    if [[ $w == */$pkg ]]; then
      echo "ERROR: $pkg should not generate itself"
      exit 1
    fi
    if [[ -f $w/BUILD.bazel ]]; then
      echo "$rule(outs=[\"${out}\"])" >> $w/BUILD
    elif  [[ -f $w/BUILD ]]; then
      echo "$rule(outs=[\"${out}\"])" >> $w/BUILD
    else
      echo cannot find build file for $w
      continue
    fi
    dozer "new_load //build:deepcopy.bzl $rule" //$w:__pkg__
  done
  echo "Deleting $out files"
  find . -iname "${out}" | xargs rm
}
update-k8s-gengo deepcopy k8s.io/code-generator/cmd/deepcopy-gen zz_generated.deepcopy.go '+k8s:deepcopy-gen='
extra_conv=($( \
  find . -name *.go | \
  xargs grep "+k8s:conversion-gen=" | \
  grep -v conversion-gen=false | grep -v cmd/conversion-gen | \
  sed -e 's|^[^=]*=||' | sort -u | $SED -e 's|k8s.io/kubernetes/|./|;s|\(^[^.]\)|./vendor/\1|'))
update-k8s-gengo conversion k8s.io/code-generator/cmd/conversion-gen zz_generated.conversion.go '+k8s:conversion-gen=' "${extra_conv[@]}"
update-k8s-gengo defaulter k8s.io/code-generator/cmd/defaulter-gen zz_generated.defaults.go '+k8s:defaulter-gen='
echo Running gazelle to cleanup any changes...
gazelle-fix

if ! which -s bazel; then
  return 0  # We should have already generated these files
fi

find . -name BUILD -or -name BUILD.bazel | xargs sed -i '' -e 's|@io_bazel_rules_go//go:def.bzl|//build:go.bzl|g'

# Add dependencies from generated files
bazel build //build:all-generated-sources
for path in $(find -H bazel-genfiles -iname *.go.deps); do
  deps="${path#bazel-genfiles/build/}" # rel/path/to/foo.go.deps
  pkg="${deps%/*}" # rel/path/to
  if [[ ! -f "$pkg/BUILD" && ! -f "$pkg/BUILD.bazel" ]]; then
    echo "$pkg: no BUILD.bazel file"
    continue
  fi
  deps=($(cat "$path"))
  deps="$(IFS=' '  ; echo "${deps[*]}")"
  dozer "add deps $deps" "//${pkg}:go_default_library"
done
