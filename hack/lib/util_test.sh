#!/usr/bin/env bash

# Copyright 2026 The Kubernetes Authors.
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

# Offline regression tests: put step, openssl, and jq on PATH, then run
# bash hack/lib/util_test.sh.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${KUBE_ROOT}/hack/lib/logging.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

for tool in step openssl jq; do
  command -v "${tool}" >/dev/null || fail "Install ${tool} and add it to PATH before running this test."
done
kube::util::test_openssl_installed
kube::util::ensure-step
installed_step="${STEP_BIN}"

# Keep temporary credentials outside the checkout and isolated from cluster runs.
work=$(mktemp -d /tmp/kube-step-util-test.XXXXXX)
work=$(cd "${work}" && pwd -P)
trap 'rm -rf "${work}"' EXIT
trap 'exit 1' HUP INT TERM

assert_json() {
  local cert=$1
  local expression=$2
  shift 2
  "${STEP_BIN}" certificate inspect --format json "${cert}" |
    jq -e "${expression}" "$@" >/dev/null ||
    fail "${cert}: expected ${expression}"
}

assert_key() {
  local cert=$1
  local key=$2
  local mode
  mode=$(stat -c %a "${key}" 2>/dev/null || stat -f %Lp "${key}")
  [[ "${mode}" == 600 ]] || fail "${key}: expected mode 600, got ${mode}"
  openssl rsa -in "${key}" -check -noout >/dev/null 2>&1 ||
    fail "${key}: invalid RSA private key"
  [[ "$(openssl x509 -in "${cert}" -noout -modulus)" == \
     "$(openssl rsa -in "${key}" -noout -modulus)" ]] ||
    fail "${cert}: certificate does not match ${key}"
  assert_json "${cert}" \
    '.subject_key_info.key_algorithm.name == "RSA" and .subject_key_info.rsa_public_key.length == 2048'
}

assert_leaf() {
  local cert=$1
  local key=$2
  local ca=$3
  assert_key "${cert}" "${key}"
  # step computes the endpoints separately; RSA generation can cross a second.
  assert_json "${cert}" \
    '.validity.length >= 157679940 and .validity.length <= 157680000 and
     .extensions.key_usage.digital_signature == true and
     .extensions.key_usage.key_encipherment == true and
     (.extensions.basic_constraints.is_ca // false) == false'
  openssl verify -CAfile "${ca}" "${cert}" >/dev/null ||
    fail "${cert}: signature does not verify against ${ca}"
}

test_ensure_step() (
  mkdir "${work}/bin" "${work}/cache" "${work}/minimal-path"
  ln -s "${installed_step}" "${work}/bin/step"
  ln -s "${installed_step}" "${work}/cache/step"
  ln -s "$(command -v mkdir)" "${work}/minimal-path/mkdir"
  # None of these paths should ever attempt a download.
  curl() { fail "ensure-step attempted a download despite an available binary"; }
  PATH="${work}/bin:${PATH}"
  kube::util::ensure-step "${work}/unused"
  [[ "${STEP_BIN}" == "${work}/bin/step" ]] || fail "Installed step was not resolved from PATH"
  [[ ! -e "${work}/unused" ]] || fail "Installed step unnecessarily created a cache"

  cd "${work}"
  PATH="bin:${PATH}"
  kube::util::ensure-step
  [[ "${STEP_BIN}" == "${work}/bin/step" ]] || fail "Relative PATH did not resolve to an absolute STEP_BIN"
  cd "${KUBE_ROOT}"
  "${STEP_BIN}" version >/dev/null

  # Hide PATH's step without hiding the one command needed to reuse the cache.
  # shellcheck disable=SC2123
  PATH="${work}/minimal-path"
  kube::util::ensure-step "${work}/cache"
  [[ "${STEP_BIN}" == "${work}/cache/step" ]] || fail "Cached step was not reused"
  "${STEP_BIN}" version >/dev/null
)

test_ensure_step
echo "PASS: installed, relative-PATH, and cached step resolution"

test_ensure_step_ppc64le() (
  local tool
  for tool in mktemp rm; do
    ln -s "$(command -v "${tool}")" "${work}/minimal-path/${tool}"
  done
  # Exercise release selection without executing a foreign-architecture binary.
  kube::util::host_arch() { echo ppc64le; }
  uname() { echo Linux; }
  curl() {
    printf '%s\n' "$@" >"${work}/ppc64le-download.args"
    return 1
  }
  # shellcheck disable=SC2123
  PATH="${work}/minimal-path"
  if kube::util::ensure-step "${work}/ppc64le-cache" 2>"${work}/ppc64le-download.log"; then
    fail "ensure-step accepted a failed ppc64le download"
  fi
)
test_ensure_step_ppc64le
grep -Fxq 'https://github.com/smallstep/cli/releases/download/v0.30.6/step_linux_0.30.6_ppc64le.tar.gz' \
  "${work}/ppc64le-download.args" || fail "Incorrect ppc64le release selected"
[[ ! -e "${work}/ppc64le-cache/step" ]] || fail "Failed ppc64le download installed a binary"
echo "PASS: Linux ppc64le release selection and download failure handling"

for purpose in client server shared; do
  case "${purpose}" in
    client) usages='"client auth"' ;;
    server) usages='"server auth"' ;;
    shared) usages='"client auth","server auth"' ;;
  esac
  kube::util::create_signing_certkey "" "${work}" "${purpose}" "${usages}" >"${work}/ca.log" 2>&1 ||
    { cat "${work}/ca.log" >&2; fail "Could not create ${purpose} CA"; }
  assert_key "${work}/${purpose}-ca.crt" "${work}/${purpose}-ca.key"
  assert_json "${work}/${purpose}-ca.crt" '.extensions.basic_constraints.is_ca == true'
done

cn='system:user "quoted" \ with spaces'
groups=('system:masters' 'group with spaces' 'group "quoted"' 'group\backslash')
groups_json=$(jq -cn --args '$ARGS.positional | sort' -- "${groups[@]}")
kube::util::create_client_certkey "" "${work}" client-ca identity "${cn}" "${groups[@]}"
assert_leaf "${work}/client-identity.crt" "${work}/client-identity.key" "${work}/client-ca.crt"
# shellcheck disable=SC2016 # These are jq variables, not shell variables.
assert_json "${work}/client-identity.crt" \
  '.subject.common_name == [$cn] and (.subject.organization | sort) == $groups and
   .extensions.extended_key_usage == {"client_auth": true} and
   (.extensions | has("subject_alt_name") | not)' \
  --arg cn "${cn}" --argjson groups "${groups_json}"
openssl verify -purpose sslclient -CAfile "${work}/client-ca.crt" "${work}/client-identity.crt" >/dev/null
if openssl verify -purpose sslserver -CAfile "${work}/client-ca.crt" "${work}/client-identity.crt" >/dev/null 2>&1; then
  fail "Client-only certificate unexpectedly permits server authentication"
fi
if openssl verify -CAfile "${work}/server-ca.crt" "${work}/client-identity.crt" >/dev/null 2>&1; then
  fail "Client certificate unexpectedly verifies against a different CA"
fi

# Omitting the username exercises the default CN and the empty groups path.
kube::util::create_client_certkey "" "${work}" client-ca default-user
assert_json "${work}/client-default-user.crt" \
  '.subject.common_name == ["default-user"] and
   (.subject.organization // []) == [] and
   (.extensions | has("subject_alt_name") | not)'
assert_leaf "${work}/client-default-user.crt" "${work}/client-default-user.key" "${work}/client-ca.crt"
echo "PASS: client identities, escaped groups, empty groups, and CA signatures"

if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
  kube::util::create_client_certkey "sudo -n" "${work}" client-ca sudo "${cn}" "${groups[@]}"
  # Only return ownership of this test's output, without widening key permissions.
  sudo -n chown "$(id -u):$(id -g)" "${work}/client-sudo.crt" "${work}/client-sudo.key"
  assert_leaf "${work}/client-sudo.crt" "${work}/client-sudo.key" "${work}/client-ca.crt"
  # shellcheck disable=SC2016 # These are jq variables, not shell variables.
  assert_json "${work}/client-sudo.crt" \
    '.subject.common_name == [$cn] and (.subject.organization | sort) == $groups and
     .extensions.extended_key_usage == {"client_auth": true} and
     (.extensions | has("subject_alt_name") | not)' \
    --arg cn "${cn}" --argjson groups "${groups_json}"
  echo "PASS: sudo client signing preserves JSON data passed through stdin"
else
  echo "SKIP: sudo client signing (noninteractive sudo unavailable)"
fi

kube::util::create_serving_certkey "" "${work}" server-ca api api.example.test \
  api.example.test localhost 127.0.0.1 ::1
assert_leaf "${work}/serving-api.crt" "${work}/serving-api.key" "${work}/server-ca.crt"
assert_json "${work}/serving-api.crt" \
  '.subject.common_name == ["api.example.test"] and
   .extensions.extended_key_usage == {"server_auth": true} and
   (.extensions.subject_alt_name.dns_names | sort) == ["api.example.test", "localhost"] and
   (.extensions.subject_alt_name.ip_addresses | sort) == ["127.0.0.1", "::1"]'
openssl verify -purpose sslserver -CAfile "${work}/server-ca.crt" "${work}/serving-api.crt" >/dev/null
if openssl verify -purpose sslclient -CAfile "${work}/server-ca.crt" "${work}/serving-api.crt" >/dev/null 2>&1; then
  fail "Server-only certificate unexpectedly permits client authentication"
fi

# Without explicit hosts, step's default CN-to-SAN injection must stay disabled.
kube::util::create_serving_certkey "" "${work}" server-ca no-hosts no-hosts.example.test
assert_leaf "${work}/serving-no-hosts.crt" "${work}/serving-no-hosts.key" "${work}/server-ca.crt"
assert_json "${work}/serving-no-hosts.crt" \
  '.subject.common_name == ["no-hosts.example.test"] and
   .extensions.extended_key_usage == {"server_auth": true} and
   (.extensions | has("subject_alt_name") | not)'

for kind in client serving; do
  if [[ "${kind}" == client ]]; then
    kube::util::create_client_certkey "" "${work}" shared-ca shared shared-user shared-group
  else
    kube::util::create_serving_certkey "" "${work}" shared-ca shared shared.example.test shared.example.test
  fi
  cert="${work}/${kind}-shared.crt"
  assert_leaf "${cert}" "${work}/${kind}-shared.key" "${work}/shared-ca.crt"
  assert_json "${cert}" '.extensions.extended_key_usage == {"client_auth": true, "server_auth": true}'
  openssl verify -purpose sslclient -CAfile "${work}/shared-ca.crt" "${cert}" >/dev/null
  openssl verify -purpose sslserver -CAfile "${work}/shared-ca.crt" "${cert}" >/dev/null
done
assert_json "${work}/client-shared.crt" '(.extensions | has("subject_alt_name") | not)'
echo "PASS: serving SANs and client-only, server-only, and shared CA usages"

old_client=$(openssl x509 -in "${work}/client-identity.crt" -noout -fingerprint -sha256)
old_server=$(openssl x509 -in "${work}/serving-api.crt" -noout -fingerprint -sha256)
# Closed stdin catches an accidental interactive overwrite prompt.
kube::util::create_client_certkey "" "${work}" client-ca identity replacement-user replacement-group </dev/null
kube::util::create_serving_certkey "" "${work}" server-ca api replacement.example.test replacement.example.test </dev/null
[[ "${old_client}" != "$(openssl x509 -in "${work}/client-identity.crt" -noout -fingerprint -sha256)" ]] ||
  fail "Client certificate was not replaced"
[[ "${old_server}" != "$(openssl x509 -in "${work}/serving-api.crt" -noout -fingerprint -sha256)" ]] ||
  fail "Serving certificate was not replaced"
assert_leaf "${work}/client-identity.crt" "${work}/client-identity.key" "${work}/client-ca.crt"
assert_leaf "${work}/serving-api.crt" "${work}/serving-api.key" "${work}/server-ca.crt"
assert_json "${work}/client-identity.crt" \
  '.subject.common_name == ["replacement-user"] and .subject.organization == ["replacement-group"]'
assert_json "${work}/serving-api.crt" \
  '.subject.common_name == ["replacement.example.test"] and
   .extensions.subject_alt_name.dns_names == ["replacement.example.test"] and
   (.extensions.subject_alt_name.ip_addresses // []) == []'
echo "PASS: noninteractive overwrite, matching RSA keys, private permissions, and five-year lifetime"

# The static template is used by callers that do not need a per-CA template.
"${STEP_BIN}" certificate create "${cn}" "${work}/template.crt" "${work}/template.key" \
  --ca "${work}/client-ca.crt" --ca-key "${work}/client-ca.key" \
  --template "${KUBE_ROOT}/hack/lib/step-client-template.json" \
  --set-file /dev/stdin --san ignored.example.test \
  --kty RSA --size 2048 --not-after 43800h --no-password --insecure \
  <<<"$(jq -cn --argjson organizations "${groups_json}" '{organizations: $organizations}')"
assert_leaf "${work}/template.crt" "${work}/template.key" "${work}/client-ca.crt"
# shellcheck disable=SC2016 # These are jq variables, not shell variables.
assert_json "${work}/template.crt" \
  '.subject.common_name == [$cn] and (.subject.organization | sort) == $groups and
   .extensions.extended_key_usage == {"client_auth": true} and
   (.extensions | has("subject_alt_name") | not)' \
  --arg cn "${cn}" --argjson groups "${groups_json}"
"${STEP_BIN}" certificate create no-groups "${work}/template.crt" "${work}/template.key" \
  --ca "${work}/client-ca.crt" --ca-key "${work}/client-ca.key" \
  --template "${KUBE_ROOT}/hack/lib/step-client-template.json" \
  --kty RSA --size 2048 --not-after 43800h --no-password --insecure --force </dev/null
assert_leaf "${work}/template.crt" "${work}/template.key" "${work}/client-ca.crt"
assert_json "${work}/template.crt" \
  '.subject.common_name == ["no-groups"] and (.subject.organization // []) == [] and
   .extensions.extended_key_usage == {"client_auth": true} and
   (.extensions | has("subject_alt_name") | not)'
echo "PASS: static client template preserves identity and suppresses SANs"

old_ca=$(openssl x509 -in "${work}/client-ca.crt" -noout -fingerprint -sha256)
kube::util::create_signing_certkey "" "${work}" client '"client auth"' >"${work}/ca.log" 2>&1 ||
  { cat "${work}/ca.log" >&2; fail "Could not replace the client CA"; }
[[ "${old_ca}" != "$(openssl x509 -in "${work}/client-ca.crt" -noout -fingerprint -sha256)" ]] ||
  fail "CA certificate was not replaced"
assert_key "${work}/client-ca.crt" "${work}/client-ca.key"
if openssl verify -CAfile "${work}/client-ca.crt" "${work}/client-identity.crt" >/dev/null 2>&1; then
  fail "Old client certificate unexpectedly verifies against the replacement CA"
fi
kube::util::create_client_certkey "" "${work}" client-ca identity </dev/null
assert_leaf "${work}/client-identity.crt" "${work}/client-identity.key" "${work}/client-ca.crt"
echo "PASS: CA replacement and signing with the replacement key"
