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

run_kubectl_config_set_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:config set)"

  kubectl config set-cluster test-cluster --server="https://does-not-work"

  # Get the api cert and add a comment to avoid flag parsing problems
  cert_data=$(echo "#Comment" && cat "${TMPDIR:-/tmp}/apiserver.crt")

  kubectl config set clusters.test-cluster.certificate-authority-data "$cert_data" --set-raw-bytes
  r_written=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  encoded=$(echo -n "$cert_data" | base64)
  kubectl config set clusters.test-cluster.certificate-authority-data "$encoded"
  e_written=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  test "$e_written" == "$r_written"

  set +o nounset
  set +o errexit
}

run_kubectl_config_set_cluster_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl config set-cluster"

  ca_file="${TMPDIR:-/tmp}/apiserver.crt"
  ca_data=$(base64 -w0 "$ca_file")

  # Set cert file
  kubectl config set-cluster test-cluster-1 --certificate-authority "$ca_file"
  expected="$ca_file"
  actual=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster-1")].cluster.certificate-authority}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Certificate authority did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  # Embed cert from file
  kubectl config set-cluster test-cluster-2 --embed-certs --certificate-authority "$ca_file"
  expected="$ca_data"
  actual=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster-2")].cluster.certificate-authority-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Certificate authority embedded from file did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  # Embed cert using process substitution
  kubectl config set-cluster test-cluster-3 --embed-certs --certificate-authority <(cat "$ca_file")
  expected="$ca_data"
  actual=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster-3")].cluster.certificate-authority-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Certificate authority embedded using process substitution did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  set +o nounset
  set +o errexit
}

run_kubectl_config_set_credentials_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl config set-credentials"

  cert_file="${TMPDIR:-/tmp}/test-client-certificate.crt"
  key_file="${TMPDIR:-/tmp}/test-client-key.crt"

  cat << EOF > "$cert_file"
-----BEGIN CERTIFICATE-----
MIIDazCCAlOgAwIBAgIUdSrvuXs0Bft9Ao/AFnC7fNBqD+owDQYJKoZIhvcNAQEL
BQAwRTELMAkGA1UEBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoM
GEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDAeFw0yMDA1MTExODMyNDJaFw0yMTA1
MTExODMyNDJaMEUxCzAJBgNVBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEw
HwYDVQQKDBhJbnRlcm5ldCBXaWRnaXRzIFB0eSBMdGQwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCrS5/kYM3TjdFw4OYMdSOys0+4aHPgtXqePWrn1pBP
W9D1yhjI/JS1/uPAWLi+1nnn0PBMCbqo+ZEsRIlSAABJE4fVbRPg+AuBPDdlCex5
QfKNi3vwp0Gy756SKTNZmIx42I9in0WeocCDliTEM2KsawCrVzuE3gwcHHkOnmLX
DEc4bajkQxcaJITG3hsJ2Cm5OBMLPwrsq/77VzOdC12r9j8+w0f7lCJfOm2ui7rm
Vl76V2Nits6U0ZrF1yzYtVQ1iWqCnOudPPf3jyc7KcSetGwozgoydkcqfUS9iMs9
2OV3v17GX6+sd8zY8tA95d/Vj6yU/l03GI9V6X9LXHSTAgMBAAGjUzBRMB0GA1Ud
DgQWBBQo2BKDxo4XI5FJDj9ZUuDst9ck7DAfBgNVHSMEGDAWgBQo2BKDxo4XI5FJ
Dj9ZUuDst9ck7DAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQAL
LjJ5k5kNdOUXFL2XhKO8w25tpkjd71vD3vKvPIN0Q5BgVmu5Bg476/WPBnSMvTVU
QYLQR5aGMLPdWuiSJnwO0BChKjOHaoCnP6D3cs2xP9hLBGoENaXMJRM4ZvFBRbES
Q6iTfq4OBPh5yCDDrjlppjhSnqaZuksmFnPFHLB/km003w8fgCD3VhmC2UFscl2K
nHaxK6uzIxisyE84ZDZYhjnPPib1wXGL8yu1dq0cbktE5+xJ2FHQkBJ6qaujkgV0
jpuWE9zk3CImFRkzPEwTF+3s5eP2XTIyWbtJGvJMmO0kHFx2PqCiAkdFldPKfrRh
M007Wf15dtkqyLNkzOxv
-----END CERTIFICATE-----
EOF

  cat << EOF > "$key_file"
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCrS5/kYM3TjdFw
4OYMdSOys0+4aHPgtXqePWrn1pBPW9D1yhjI/JS1/uPAWLi+1nnn0PBMCbqo+ZEs
RIlSAABJE4fVbRPg+AuBPDdlCex5QfKNi3vwp0Gy756SKTNZmIx42I9in0WeocCD
liTEM2KsawCrVzuE3gwcHHkOnmLXDEc4bajkQxcaJITG3hsJ2Cm5OBMLPwrsq/77
VzOdC12r9j8+w0f7lCJfOm2ui7rmVl76V2Nits6U0ZrF1yzYtVQ1iWqCnOudPPf3
jyc7KcSetGwozgoydkcqfUS9iMs92OV3v17GX6+sd8zY8tA95d/Vj6yU/l03GI9V
6X9LXHSTAgMBAAECggEASKlTsfS+WrcV2OQNscse0XbuojLstK1GzkkPSDjkDkXM
ZfbMfLVn/6uXwMfh1lH0dDlVNWwLGhKDWlvYREhr1pPKUuZqQEv31WJNvTZwcR9g
XFqGwJayb8zlXurLNX5YWArFB/i394p1t1vBTNjfSnQ5XHUscjgeuu35DBJzqvSA
mh1nDUBFIMC4ruPzD7OMI7rXBN02PbfTBpoL8MGoesW0S/BFQciMS07rELL2hFih
zZPJE+E6h3GMoSPAzqx+ubSkhMbxb/xQQSw4gfp8zii2SMuq8Ith9xaGhdj1TuaX
NqZahgSRe7bbgYyHWHXYbd7gQa4fxxoT7Qyh0cSKwQKBgQDZP0yYcFfNADFfnw62
eWgNHnfPsOTYa/gAXdKgueUekT+linGvAUQtdXxnLoVMiLXSVxaJbW1rAP0avJt8
EJiTdtTXkKqTqwtfbUP/bmxjYNExvGfIOpJbDsCX+kwpCoWzHupXeQlZn7UDEmKR
l0DWUVzqNnhO0WWaM9J4MD4BjwKBgQDJ2elu8h7V/U/0dr2e4GG1/Xoa1nIcct8C
rn0cG2GJ6UxN9Rr/brrLZuBewlhul9VjGtcmD7ZvEEsR4EUHUguvheAg90rpAqSE
c6LOYdGAsUa21iuVLPKPMFwd4MhtrP2JcwHO+oqlUK4939TlZEtyiMWsMJGuugh1
nrudZ9LSvQKBgBFG83R8Gr928HZGVAk3BotkjOq7irebfpGo5INbxVj0/DbSF9Bv
LVjgKxCZpog7pxofSu+LAFSuM3LY5RSszTWNEchC/Q3ZYIIqUmoSAhS1Mm3eKfLG
lbUgKzjq8vuglpl0L/bc7V1vUhn4cFZbzRA+UEFgK5k5Ffd5f5eHXqcJAoGBAJmA
hVwg9sBHfnlrn3JmMwiCdkxYjrkBxoS0i2JHlFqbt7KFVn2wCI/McY6+fx/Dibxv
WfSQ+Gzn2B8FDZmulEJsLfED/szKfLBZfBM1Imya5CsBHm24m9G2tibmnaWCa+EO
O+7aa3uiqo9VXAMCzbmRN7plyTQ2N16zUvw2S4aFAoGARoPL2cJ+7HHcnc4SzKm4
Su5nLwVPj+IJUwI5SRsRdnUKqo4gaX54p/u/TlA6fm7esRqPu5LK0oIyItVP7wmT
nUCUFtnE53Rm2QT+BlYg7CewkaRiUHgQR31RDsQP8XtQouy0Si2jG53QLtBm+b7D
zpqQAUELuiSK67vTd+D96ss=
-----END PRIVATE KEY-----
EOF

  cert_data=$(base64 -w0 "$cert_file")
  key_data=$(base64 -w0 "$key_file")

  # Set client certificate and client key files
  kubectl config set-credentials user1 --client-certificate="$cert_file" --client-key="$key_file"
  expected="$cert_file"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user1")].user.client-certificate}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client certificate did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi
  expected="$key_file"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user1")].user.client-key}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client key did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  # Embed client certificate and client key from files
  kubectl config set-credentials user2 --client-certificate="$cert_file" --client-key="$key_file" --embed-certs=true
  expected="$cert_data"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user2")].user.client-certificate-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client certificate data embedded from file did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi
  expected="$key_data"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user2")].user.client-key-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client key data embedded from file did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  # Embed client certificate and client key using process substitution
  kubectl config set-credentials user3 --client-certificate=<(cat "$cert_file") --client-key=<(cat "$key_file") --embed-certs=true
  expected="$cert_data"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user3")].user.client-certificate-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client certificate data embedded using process substitution did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi
  expected="$key_data"
  actual=$(kubectl config view --raw -o jsonpath='{.users[?(@.name == "user3")].user.client-key-data}')
  if [ "$expected" != "$actual" ]; then
    kube::log::error "Client key data embedded using process substitution did not match the expected value (expected=$expected, actual=$actual)"
    exit 1
  fi

  set +o nounset
  set +o errexit
}

run_client_config_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing client config"

  # Command
  # Pre-condition: kubeconfig "missing" is not a file or directory
  output_message=$(! kubectl get pod --context="" --kubeconfig=missing 2>&1)
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: kubeconfig "missing" is not a file or directory
  # Command
  output_message=$(! kubectl get pod --user="" --kubeconfig=missing 2>&1)
  # Post-condition: --user contains a valid / empty value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"
  # Command
  output_message=$(! kubectl get pod --cluster="" --kubeconfig=missing 2>&1)
  # Post-condition: --cluster contains a "valid" value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: context "missing-context" does not exist
  # Command
  output_message=$(! kubectl get pod --context="missing-context" 2>&1)
  kube::test::if_has_string "${output_message}" 'context was not found for specified context: missing-context'
  # Post-condition: invalid or missing context returns error

  # Pre-condition: cluster "missing-cluster" does not exist
  # Command
  output_message=$(! kubectl get pod --cluster="missing-cluster" 2>&1)
  kube::test::if_has_string "${output_message}" 'no server found for cluster "missing-cluster"'
  # Post-condition: invalid or missing cluster returns error

  # Pre-condition: user "missing-user" does not exist
  # Command
  output_message=$(! kubectl get pod --user="missing-user" 2>&1)
  kube::test::if_has_string "${output_message}" 'auth info "missing-user" does not exist'
  # Post-condition: invalid or missing user returns error

  # test invalid config
  kubectl config view | sed -E "s/apiVersion: .*/apiVersion: v-1/g" > "${TMPDIR:-/tmp}"/newconfig.yaml
  output_message=$(! "${THIS_PLATFORM_BIN}/kubectl" get pods --context="" --user="" --kubeconfig="${TMPDIR:-/tmp}"/newconfig.yaml 2>&1)
  kube::test::if_has_string "${output_message}" "error loading config file"

  output_message=$(! kubectl get pod --kubeconfig=missing-config 2>&1)
  kube::test::if_has_string "${output_message}" 'no such file or directory'

  set +o nounset
  set +o errexit
}

run_kubeconfig_impersonate_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing config with impersonation"

  # copy the existing kubeconfig over and add a new user entry for the admin user impersonating a different user.
  kubectl config view --raw > "${TMPDIR:-/tmp}"/impersonateconfig.yaml
  cat << EOF >> "${TMPDIR:-/tmp}"/impersonateconfig.yaml
users:
- name: admin-as-userb
  user:
    # token defined in hack/testdata/auth-tokens.csv
    token: admin-token
    # impersonated user
    as: userb
    as-uid: abc123
    as-groups:
    - group2
    - group1
    as-user-extra:
      foo:
      - bar
      - baz
- name: as-uid-without-as
  user:
    # token defined in hack/testdata/auth-tokens.csv
    token: admin-token
    # impersonated uid
    as-uid: abc123
EOF

  kubectl create -f hack/testdata/csr.yml --kubeconfig "${TMPDIR:-/tmp}"/impersonateconfig.yaml --user admin-as-userb
  kube::test::get_object_assert 'csr/foo' '{{.spec.username}}' 'userb'
  kube::test::get_object_assert 'csr/foo' '{{.spec.uid}}' 'abc123'
  kube::test::get_object_assert 'csr/foo' '{{range .spec.groups}}{{.}} {{end}}' 'group2 group1 system:authenticated '
  kube::test::get_object_assert 'csr/foo' '{{len .spec.extra}}' '1'
  kube::test::get_object_assert 'csr/foo' '{{range .spec.extra.foo}}{{.}} {{end}}' 'bar baz '
  kubectl delete -f hack/testdata/csr.yml

  output_message=$(! kubectl get pods --kubeconfig "${TMPDIR:-/tmp}"/impersonateconfig.yaml --user as-uid-without-as 2>&1)
  kube::test::if_has_string "${output_message}" 'without impersonating a user'

  set +o nounset
  set +o errexit
}
