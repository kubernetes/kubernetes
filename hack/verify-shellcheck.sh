#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"


# disabled lints
disabled=(
  # this lint dissalows non-constant source, which we use extensively
  1090
  # this lint prefers command -v to which, they are not the same
  2230
)
# comma separate for passing to shellcheck
join_by() {
  local IFS="$1";
  shift;
  echo "$*";
}
SHELLCHECK_DISABLED="$(join_by , "${disabled[@]}")"
readonly SHELLCHECK_DISABLED

if ! which shellcheck > /dev/null; then
  echo 'Can not find shellcheck, please install shellcheck to run this script'
  echo 'see: https://github.com/koalaman/shellcheck#installing'
  # TODO(bentheelder): we should discuss how to better handle this
  exit 1
fi

cd "${KUBE_ROOT}"

# find all shell scripts excluding ./_* and ./vendor*
all_shell_scripts=()
while IFS=$'\n' read -r script;
  do all_shell_scripts+=("$script");
done < <(find . -name "*.sh" \
  -not \( \
    -path ./_\*      -o \
    -path ./vendor\*    \
  \))

# make sure known failures are sorted
failure_file="${KUBE_ROOT}/hack/.shellcheck_failures"
if ! diff -u "${failure_file}" <(LC_ALL=C sort "${failure_file}"); then
  {
    echo
    echo "hack/.shellcheck_failures is not in alphabetical order. Please sort it:"
    echo
    echo "  LC_ALL=C sort -o hack/.shellcheck_failures hack/.shellcheck_failures"
    echo
  } >&2
  false
fi

# load known failure files
failing_files=()
while IFS=$'\n' read -r script;
  do failing_files+=("$script");
done < <(cat "${failure_file}")

# TODO(bentheelder): we should probably move this and the copy in verify-golint.sh
# to one of the bash libs
array_contains () {
  local seeking=$1; shift # shift will iterate through the array
  local in=1 # in holds the exit status for the function
  for element; do
    if [[ "$element" == "$seeking" ]]; then
      in=0 # set in to 0 since we found it
      break
    fi
  done
  return $in
}

# lint each script, tracking failures
errors=()
not_failing=()
for f in "${all_shell_scripts[@]}"; do
  set +o errexit
  failedLint=$(shellcheck --exclude="${SHELLCHECK_DISABLED}" "${f}")
  set -o errexit
  array_contains "${f}" "${failing_files[@]}" && in_failing=$? || in_failing=$?
  if [[ -n "${failedLint}" ]] && [[ "${in_failing}" -ne "0" ]]; then
    errors+=( "${failedLint}" )
  fi
  if [[ -z "${failedLint}" ]] && [[ "${in_failing}" -eq "0" ]]; then
    not_failing+=( "${f}" )
  fi
done

# Check to be sure all the packages that should pass lint are.
if [ ${#errors[@]} -eq 0 ]; then
  echo 'Congratulations!  All shell files have been linted.'
else
  {
    echo "Errors from shellcheck:"
    for err in "${errors[@]}"; do
      echo "$err"
    done
    echo
    echo 'Please review the above warnings. You can test via "./hack/verify-shellcheck"'
    echo 'If the above warnings do not make sense, you can exempt this package from shellcheck'
    echo 'checking by adding it to hack/.shellcheck_failures (if your reviewer is okay with it).'
    echo
  } >&2
  false
fi

if [[ ${#not_failing[@]} -gt 0 ]]; then
  {
    echo "Some packages in hack/.shellcheck_failures are passing shellcheck. Please remove them."
    echo
    for f in "${not_failing[@]}"; do
      echo "  $f"
    done
    echo
  } >&2
  false
fi

# Check that all failing_packages actually still exist
gone=()
for f in "${failing_files[@]}"; do
  array_contains "$f" "${all_shell_scripts[@]}" || gone+=( "$f" )
done

if [[ ${#gone[@]} -gt 0 ]]; then
  {
    echo "Some files in hack/.shellcheck_failures do not exist anymore. Please remove them."
    echo
    for f in "${gone[@]}"; do
      echo "  $f"
    done
    echo
  } >&2
  false
fi
