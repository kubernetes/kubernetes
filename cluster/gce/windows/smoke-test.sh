#!/bin/bash

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

# A small smoke test to run against a just-deployed kube-up cluster with Windows
# nodes. Performs checks such as:
#   1) Verifying that all Windows nodes have status Ready.
#   2) Verifying that no system pods are attempting to run on Windows nodes.
#   3) Verifying pairwise connectivity between most of the following: Linux
#      pods, Windows pods, K8s services, and the Internet.
#   4) Verifying that basic DNS resolution works in Windows pods.
#
# This script assumes that it is run from the root of the kubernetes repository.
#
# TODOs:
#   - Implement the node-to-pod checks.
#   - Capture stdout for each command to a file and only print it when the test
#     fails.
#   - Move copy-pasted code into reusable functions.
#   - Continue running all checks after one fails.
#   - Test service connectivity by running a test pod with an http server and
#     exposing it as a service (rather than curl-ing from existing system
#     services that don't serve http requests).
#   - Add test retries for transient errors, such as:
#     "error: unable to upgrade connection: Authorization error
#     (user=kube-apiserver, verb=create, resource=nodes, subresource=proxy)"

# Override this to use a different kubectl binary.
kubectl=kubectl
linux_deployment_timeout=60
windows_deployment_timeout=600
output_file=/tmp/k8s-smoke-test.out

function check_windows_nodes_are_ready {
  # kubectl filtering is the worst.
  statuses=$(${kubectl} get nodes -l kubernetes.io/os=windows \
    -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}')
  for status in $statuses; do
    if [[ $status == "False" ]]; then
      echo "ERROR: some Windows node has status != Ready"
      echo "kubectl get nodes -l kubernetes.io/os=windows"
      ${kubectl} get nodes -l kubernetes.io/os=windows
      exit 1
    fi
  done
  echo "Verified that all Windows nodes have status Ready"
}

function untaint_windows_nodes {
  # Untaint the windows nodes to allow test workloads without tolerations to be
  # scheduled onto them.
  WINDOWS_NODES=$(${kubectl} get nodes -l kubernetes.io/os=windows -o name)
  for node in $WINDOWS_NODES; do
    ${kubectl} taint node "$node" node.kubernetes.io/os:NoSchedule-
  done
}

function check_no_system_pods_on_windows_nodes {
  windows_system_pods=$(${kubectl} get pods --namespace kube-system \
    -o wide | grep -E "Pending|windows" | wc -w)
  if [[ $windows_system_pods -ne 0 ]]; then
    echo "ERROR: there are kube-system pods trying to run on Windows nodes"
    echo "kubectl get pods --namespace kube-system -o wide"
    ${kubectl} get pods --namespace kube-system -o wide
    exit 1
  fi
  echo "Verified that all system pods are running on Linux nodes"
}

linux_webserver_deployment=linux-nginx
linux_webserver_pod_label=nginx
linux_webserver_replicas=1

function deploy_linux_webserver_pod {
  echo "Writing example deployment to $linux_webserver_deployment.yaml"
  cat <<EOF > $linux_webserver_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $linux_webserver_deployment
  labels:
    app: $linux_webserver_pod_label
spec:
  replicas: $linux_webserver_replicas
  selector:
    matchLabels:
      app: $linux_webserver_pod_label
  template:
    metadata:
      labels:
        app: $linux_webserver_pod_label
    spec:
      containers:
      - name: nginx
        image: nginx:1.7.9
      nodeSelector:
        kubernetes.io/os: linux
EOF

  if ! ${kubectl} create -f $linux_webserver_deployment.yaml; then
    echo "kubectl create -f $linux_webserver_deployment.yaml failed"
    exit 1
  fi

  timeout=$linux_deployment_timeout
  while [[ $timeout -gt 0 ]]; do
    echo "Waiting for $linux_webserver_replicas Linux $linux_webserver_pod_label pods to become Ready"
    statuses=$(${kubectl} get pods -l app=$linux_webserver_pod_label \
      -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' \
      | grep "True" | wc -w)
    if [[ $statuses -eq $linux_webserver_replicas ]]; then
      break
    else
      sleep 10
      (( timeout=timeout-10 ))
    fi
  done

  if [[ $timeout -gt 0 ]]; then
    echo "All $linux_webserver_pod_label pods became Ready"
  else
    echo "ERROR: Not all $linux_webserver_pod_label pods became Ready"
    echo "kubectl get pods -l app=$linux_webserver_pod_label"
    ${kubectl} get pods -l app=$linux_webserver_pod_label
    cleanup_deployments
    exit 1
  fi
}

# Returns the IP address of an arbitrary Linux webserver pod.
function get_linux_webserver_pod_ip {
  $kubectl get pods -l app="$linux_webserver_pod_label" \
    -o jsonpath='{.items[0].status.podIP}'
}

function undeploy_linux_webserver_pod {
  ${kubectl} delete deployment $linux_webserver_deployment
}

linux_command_deployment=linux-ubuntu
linux_command_pod_label=ubuntu
linux_command_replicas=1

function deploy_linux_command_pod {
  echo "Writing example deployment to $linux_command_deployment.yaml"
  cat <<EOF > $linux_command_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $linux_command_deployment
  labels:
    app: $linux_command_pod_label
spec:
  replicas: $linux_command_replicas
  selector:
    matchLabels:
      app: $linux_command_pod_label
  template:
    metadata:
      labels:
        app: $linux_command_pod_label
    spec:
      containers:
      - name: ubuntu
        image: ubuntu
        command: ["sleep", "123456"]
      nodeSelector:
        kubernetes.io/os: linux
EOF

  if ! ${kubectl} create -f $linux_command_deployment.yaml; then
    echo "kubectl create -f $linux_command_deployment.yaml failed"
    exit 1
  fi

  timeout=$linux_deployment_timeout
  while [[ $timeout -gt 0 ]]; do
    echo "Waiting for $linux_command_replicas Linux $linux_command_pod_label pods to become Ready"
    statuses=$(${kubectl} get pods -l app="$linux_command_pod_label" \
      -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' \
      | grep "True" | wc -w)
    if [[ $statuses -eq $linux_command_replicas ]]; then
      break
    else
      sleep 10
      (( timeout=timeout-10 ))
    fi
  done

  if [[ $timeout -gt 0 ]]; then
    echo "All $linux_command_pod_label pods became Ready"
  else
    echo "ERROR: Not all $linux_command_pod_label pods became Ready"
    echo "kubectl get pods -l app=$linux_command_pod_label"
    ${kubectl} get pods -l app="$linux_command_pod_label"
    cleanup_deployments
    exit 1
  fi
}

# Returns the name of an arbitrary Linux command pod.
function get_linux_command_pod_name {
  $kubectl get pods -l app="$linux_command_pod_label" \
    -o jsonpath='{.items[0].metadata.name}'
}

# Installs test executables (ping, curl) in the Linux command pod.
# NOTE: this assumes that there is only one Linux "command pod".
# TODO(pjh): fix this.
function prepare_linux_command_pod {
  local linux_command_pod
  linux_command_pod="$(get_linux_command_pod_name)"

  echo "Installing test utilities in Linux command pod, may take a minute"
  $kubectl exec "$linux_command_pod" -- apt-get update > /dev/null
  $kubectl exec "$linux_command_pod" -- \
    apt-get install -y iputils-ping curl > /dev/null
}

function undeploy_linux_command_pod {
  ${kubectl} delete deployment $linux_command_deployment
}

windows_webserver_deployment=windows-agnhost
windows_webserver_pod_label=agnhost
# The default port for 'agnhost serve-hostname'. The documentation says that
# this can be changed but the --port arg does not seem to work.
windows_webserver_port=9376
windows_webserver_replicas=1

function deploy_windows_webserver_pod {
  echo "Writing example deployment to $windows_webserver_deployment.yaml"
  cat <<EOF > $windows_webserver_deployment.yaml
# A multi-arch Windows container that runs an HTTP server on port
# $windows_webserver_port that serves the container's hostname.
#   curl -s http://<pod_ip>:$windows_webserver_port
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $windows_webserver_deployment
  labels:
    app: $windows_webserver_pod_label
spec:
  replicas: $windows_webserver_replicas
  selector:
    matchLabels:
      app: $windows_webserver_pod_label
  template:
    metadata:
      labels:
        app: $windows_webserver_pod_label
    spec:
      containers:
      - name: agnhost
        image: e2eteam/agnhost:2.26
        args:
        - serve-hostname
      nodeSelector:
        kubernetes.io/os: windows
      tolerations:
      - effect: NoSchedule
        key: node.kubernetes.io/os
        operator: Equal
        value: windows
EOF

  if ! ${kubectl} create -f $windows_webserver_deployment.yaml; then
    echo "kubectl create -f $windows_webserver_deployment.yaml failed"
    exit 1
  fi

  timeout=$windows_deployment_timeout
  while [[ $timeout -gt 0 ]]; do
    echo "Waiting for $windows_webserver_replicas Windows $windows_webserver_pod_label pods to become Ready"
    statuses=$(${kubectl} get pods -l app=$windows_webserver_pod_label \
      -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' \
      | grep "True" | wc -w)
    if [[ $statuses -eq $windows_webserver_replicas ]]; then
      break
    else
      sleep 10
      (( timeout=timeout-10 ))
    fi
  done

  if [[ $timeout -gt 0 ]]; then
    echo "All $windows_webserver_pod_label pods became Ready"
  else
    echo "ERROR: Not all $windows_webserver_pod_label pods became Ready"
    echo "kubectl get pods -l app=$windows_webserver_pod_label"
    ${kubectl} get pods -l app=$windows_webserver_pod_label
    cleanup_deployments
    exit 1
  fi
}

function get_windows_webserver_pod_ip {
  ${kubectl} get pods -l app="$windows_webserver_pod_label" \
    -o jsonpath='{.items[0].status.podIP}'
}

function undeploy_windows_webserver_pod {
  ${kubectl} delete deployment "$windows_webserver_deployment"
}

windows_command_deployment=windows-powershell
windows_command_pod_label=powershell
windows_command_replicas=1

# Deploys a multi-arch Windows pod capable of running PowerShell.
function deploy_windows_command_pod {
  echo "Writing example deployment to $windows_command_deployment.yaml"
  cat <<EOF > $windows_command_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $windows_command_deployment
  labels:
    app: $windows_command_pod_label
spec:
  replicas: $windows_command_replicas
  selector:
    matchLabels:
      app: $windows_command_pod_label
  template:
    metadata:
      labels:
        app: $windows_command_pod_label
    spec:
      containers:
      - name: pause-win
        image: registry.k8s.io/pause:3.10
      nodeSelector:
        kubernetes.io/os: windows
      tolerations:
      - effect: NoSchedule
        key: node.kubernetes.io/os
        operator: Equal
        value: windows
EOF

  if ! ${kubectl} create -f $windows_command_deployment.yaml; then
    echo "kubectl create -f $windows_command_deployment.yaml failed"
    exit 1
  fi

  timeout=$windows_deployment_timeout
  while [[ $timeout -gt 0 ]]; do
    echo "Waiting for $windows_command_replicas Windows $windows_command_pod_label pods to become Ready"
    statuses=$(${kubectl} get pods -l app=$windows_command_pod_label \
      -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' \
      | grep "True" | wc -w)
    if [[ $statuses -eq $windows_command_replicas ]]; then
      break
    else
      sleep 10
      (( timeout=timeout-10 ))
    fi
  done

  if [[ $timeout -gt 0 ]]; then
    echo "All $windows_command_pod_label pods became Ready"
  else
    echo "ERROR: Not all $windows_command_pod_label pods became Ready"
    echo "kubectl get pods -l app=$windows_command_pod_label"
    ${kubectl} get pods -l app=$windows_command_pod_label
    cleanup_deployments
    exit 1
  fi
}

function get_windows_command_pod_name {
  $kubectl get pods -l app="$windows_command_pod_label" \
    -o jsonpath='{.items[0].metadata.name}'
}

function undeploy_windows_command_pod {
  ${kubectl} delete deployment "$windows_command_deployment"
}

function test_linux_node_to_linux_pod {
  echo "TODO: ${FUNCNAME[0]}"
}

function test_linux_node_to_windows_pod {
  echo "TODO: ${FUNCNAME[0]}"
}

function test_linux_pod_to_linux_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local linux_command_pod
  linux_command_pod="$(get_linux_command_pod_name)"
  local linux_webserver_pod_ip
  linux_webserver_pod_ip="$(get_linux_webserver_pod_ip)"

  if ! $kubectl exec "$linux_command_pod" -- curl -s -m 20 \
      "http://$linux_webserver_pod_ip" &> $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

# TODO(pjh): this test flakily fails on brand-new clusters, not sure why.
# % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
#                                Dload  Upload   Total   Spent    Left  Speed
# 0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
# curl: (6) Could not resolve host:
# command terminated with exit code 6
function test_linux_pod_to_windows_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local linux_command_pod
  linux_command_pod="$(get_linux_command_pod_name)"
  local windows_webserver_pod_ip
  windows_webserver_pod_ip="$(get_windows_webserver_pod_ip)"

  if ! $kubectl exec "$linux_command_pod" -- curl -s -m 20 \
      "http://$windows_webserver_pod_ip:$windows_webserver_port" &> $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    echo "This test seems to be flaky. TODO(pjh): investigate."
    exit 1
  fi
}

function test_linux_pod_to_k8s_service {
  echo "TEST: ${FUNCNAME[0]}"
  local linux_command_pod
  linux_command_pod="$(get_linux_command_pod_name)"
  local service="metrics-server"
  local service_ip
  service_ip=$($kubectl get service --namespace kube-system $service \
    -o jsonpath='{.spec.clusterIP}')
  local service_port
  service_port=$($kubectl get service --namespace kube-system $service \
    -o jsonpath='{.spec.ports[?(@.protocol=="TCP")].port}')
  echo "curl-ing $service address from Linux pod: $service_ip:$service_port"

  # curl-ing the metrics-server service downloads 14 bytes of unprintable binary
  # data and sets a return code of success (0).
  if ! $kubectl exec "$linux_command_pod" -- \
      curl -s -m 20 --insecure "https://$service_ip:$service_port" &> $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_windows_node_to_linux_pod {
  echo "TODO: ${FUNCNAME[0]}"
}

function test_windows_node_to_windows_pod {
  echo "TODO: ${FUNCNAME[0]}"
}

# TODO(pjh): this test failed for me once with
#   error: unable to upgrade connection: container not found ("nettest")
# Maybe the container crashed for some reason? Investigate if it happens more.
#
# TODO(pjh): another one-time failure:
#   error: unable to upgrade connection: Authorization error
#   (user=kube-apiserver, verb=create, resource=nodes, subresource=proxy)
function test_windows_pod_to_linux_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"
  local linux_webserver_pod_ip
  linux_webserver_pod_ip="$(get_linux_webserver_pod_ip)"

  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "curl -UseBasicParsing http://$linux_webserver_pod_ip" > \
      $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_windows_pod_to_windows_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"
  local windows_webserver_pod_ip
  windows_webserver_pod_ip="$(get_windows_webserver_pod_ip)"

  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "curl -UseBasicParsing http://$windows_webserver_pod_ip:$windows_webserver_port" \
      > $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_windows_pod_to_internet {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"
  # A stable (hopefully) HTTP server provided by Cloudflare. If this ever stops
  # working, we can request from 8.8.8.8 (Google DNS) using https instead.
  local internet_ip="1.1.1.1"

  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "curl -UseBasicParsing http://$internet_ip" > $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_windows_pod_to_k8s_service {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"
  local service="metrics-server"
  local service_ip
  service_ip=$($kubectl get service --namespace kube-system $service \
    -o jsonpath='{.spec.clusterIP}')
  local service_port
  service_port=$($kubectl get service --namespace kube-system $service \
    -o jsonpath='{.spec.ports[?(@.protocol=="TCP")].port}')
  local service_address="$service_ip:$service_port"

  echo "curl-ing $service address from Windows pod: $service_address"
  # curl-ing the metrics-server service results in a ServerProtocolViolation
  # ("The server committed a protocol violation. Section=ResponseStatusLine")
  # exception. Since we don't care about what the metrics-server actually gives
  # back to us, just that we can reach it, we check that we get the expected
  # exception code and not some other exception code.
  # TODO: it might be less fragile to check that we don't get the "Unable to
  # connect to the remote server" exception code (2) instead of specifically
  # expecting the protocol-violation exception code (11).
  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "\$result = try { \`
         curl -UseBasicParsing http://$service_address -ErrorAction Stop \`
       } catch [System.Net.WebException] { \`
         \$_ \`
       }; \`
       if ([int]\$result.Exception.Status -eq 11) { \`
         Write-Host \"curl $service_address got expected exception\"
         exit 0 \`
       } else { \`
         Write-Host \"curl $service_address got unexpected result/exception: \$result\"
         exit 1 \`
       }" > $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_kube_dns_in_windows_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"
  local service="kube-dns"
  local service_ip
  service_ip=$($kubectl get service --namespace kube-system $service \
    -o jsonpath='{.spec.clusterIP}')

  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "Resolve-DnsName www.bing.com -server $service_ip" > $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function test_dns_just_works_in_windows_pod {
  echo "TEST: ${FUNCNAME[0]}"
  local windows_command_pod
  windows_command_pod="$(get_windows_command_pod_name)"

  if ! $kubectl exec "$windows_command_pod" -- powershell.exe \
      "curl -UseBasicParsing http://www.bing.com" > $output_file; then
    cleanup_deployments
    echo "Failing output: $(cat $output_file)"
    echo "FAILED: ${FUNCNAME[0]}"
    exit 1
  fi
}

function cleanup_deployments {
  undeploy_linux_webserver_pod
  undeploy_linux_command_pod
  undeploy_windows_webserver_pod
  undeploy_windows_command_pod
}

check_windows_nodes_are_ready
untaint_windows_nodes
check_no_system_pods_on_windows_nodes

deploy_linux_webserver_pod
deploy_linux_command_pod
deploy_windows_webserver_pod
deploy_windows_command_pod
prepare_linux_command_pod
echo ""

test_linux_node_to_linux_pod
test_linux_node_to_windows_pod
test_linux_pod_to_linux_pod
test_linux_pod_to_windows_pod
test_linux_pod_to_k8s_service

# Note: test_windows_node_to_k8s_service is not supported at this time.
# https://docs.microsoft.com/en-us/virtualization/windowscontainers/kubernetes/common-problems#my-windows-node-cannot-access-my-services-using-the-service-ip
test_windows_node_to_linux_pod
test_windows_node_to_windows_pod
test_windows_pod_to_linux_pod
test_windows_pod_to_windows_pod
test_windows_pod_to_internet
test_windows_pod_to_k8s_service
test_kube_dns_in_windows_pod
test_dns_just_works_in_windows_pod
echo ""

cleanup_deployments
echo "All tests passed!"
exit 0
