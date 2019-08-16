#!/usr/bin/env bash
# Copyright 2019 The Kubernetes Authors.
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
cd "${KUBE_ROOT}"

all_e2e_files=()
# NOTE: This checks e2e test code without the e2e framework which contains Expect().To(HaveOccurred())
kube::util::read-array all_e2e_files < <(find test/e2e{,_node,_kubeadm} -name '*.go' | grep -v 'test/e2e/framework/')
errors_expect_no_error=()
for file in "${all_e2e_files[@]}"
do
    if grep -E "Expect\(.*\)\.(NotTo|ToNot)\(.*HaveOccurred\(\)" "${file}" > /dev/null
    then
        errors_expect_no_error+=( "${file}" )
    fi
done

errors_expect_error=()
for file in "${all_e2e_files[@]}"
do
    if grep "Expect(.*)\.To(.*HaveOccurred()" "${file}" > /dev/null
    then
        errors_expect_error+=( "${file}" )
    fi
done

errors_expect_equal=()
for file in "${all_e2e_files[@]}"
do
    if grep -E "Expect\(.*\)\.To\((gomega\.Equal|Equal)" "${file}" > /dev/null
    then
        errors_expect_equal+=( "${file}" )
    fi
done

if [ ${#errors_expect_no_error[@]} -ne 0 ]; then
  {
    echo "Errors:"
    for err in "${errors_expect_no_error[@]}"; do
      echo "$err"
    done
    echo
    echo 'The above files need to use framework.ExpectNoError(err) instead of '
    echo 'Expect(err).NotTo(HaveOccurred()) or gomega.Expect(err).NotTo(gomega.HaveOccurred())'
    echo
  } >&2
  exit 1
fi

if [ ${#errors_expect_error[@]} -ne 0 ]; then
  {
    echo "Errors:"
    for err in "${errors_expect_error[@]}"; do
      echo "$err"
    done
    echo
    echo 'The above files need to use framework.ExpectError(err) instead of '
    echo 'Expect(err).To(HaveOccurred()) or gomega.Expect(err).To(gomega.HaveOccurred())'
    echo
  } >&2
  exit 1
fi

if [ ${#errors_expect_equal[@]} -ne 0 ]; then
  {
    echo "Errors:"
    for err in "${errors_expect_equal[@]}"; do
      echo "$err"
    done
    echo
    echo 'The above files need to use framework.ExpectEqual(foo, bar) instead of '
    echo 'Expect(foo).To(Equal(bar)) or gomega.Expect(foo).To(gomega.Equal(bar))'
    echo
  } >&2
  exit 1
fi

echo 'Congratulations!  All e2e test source files are valid.'
