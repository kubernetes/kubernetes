#!/bin/sh

set -o nounset
set -o errexit
set -o pipefail

# This script tests the kube-proxy image without actually using it as
# part of the infrastructure of a cluster. It is intended to be copied
# to the kubernetes-tests image for use in CI and should have no
# dependencies beyond oc and basic shell stuff.

# There is no good way to "properly" test the kube-proxy image in
# OpenShift CI, because it is only used as a dependency of third-party
# software (e.g. Calico); no fully-RH-supported configuration uses it.
#
# However, since we don't apply any kube-proxy-specific patches to our
# tree, we can assume that it *mostly* works, since we are building
# from sources that passed upstream testing. This script is just to
# confirm that our build is not somehow completely broken (e.g.
# immediate segfault due to a bad build environment).

if [[ -z "${KUBE_PROXY_IMAGE}" ]]; then
    echo "KUBE_PROXY_IMAGE not set" 1>&2
    exit 1
fi

TMPDIR=$(mktemp --tmpdir -d kube-proxy.XXXXXX)
function cleanup() {
    oc delete namespace kube-proxy-test || true
    oc delete clusterrole kube-proxy-test || true
    oc delete clusterrolebinding kube-proxy-test || true
    rm -rf "${TMPDIR}"
}
trap "cleanup" EXIT

function indent() {
    sed -e 's/^/  /' "$@"
    echo ""
}

# Decide what kube-proxy mode to use.
# (jsonpath expression copied from types_cluster_version.go)
OCP_VERSION=$(oc get clusterversion version -o jsonpath='{.status.history[?(@.state=="Completed")].version}')
case "${OCP_VERSION}" in
    4.17.*|4.18.*)
        # 4.17 and 4.18 always use RHEL 9 (and nftables mode was still alpha in 4.17), so
        # use iptables mode
        PROXY_MODE="iptables"
        ;;
    *)
        # 4.19 and later may use RHEL 10, so use nftables mode
        PROXY_MODE="nftables"
        ;;
esac

echo "Setting up Namespace and RBAC"
oc create -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: kube-proxy-test
  labels:
    pod-security.kubernetes.io/enforce: privileged
    pod-security.kubernetes.io/audit: privileged
    pod-security.kubernetes.io/warn: privileged
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kube-proxy-test
rules:
- apiGroups: [""]
  resources:
  - namespaces
  - endpoints
  - services
  - pods
  - nodes
  verbs:
  - get
  - list
  - watch
- apiGroups: ["discovery.k8s.io"]
  resources:
  - endpointslices
  verbs:
  - get
  - list
  - watch
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kube-proxy-test
  namespace: kube-proxy-test
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kube-proxy-test
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kube-proxy-test
subjects:
- kind: ServiceAccount
  name: kube-proxy-test
  namespace: kube-proxy-test
EOF
echo ""

# We run kube-proxy in a pod-network pod, so that it can create rules
# in that pod's network namespace without interfering with
# ovn-kubernetes in the host network namespace.
#
# We need to manually set all of the conntrack values to 0 so it won't
# try to set the sysctls (which would fail). This is the most fragile
# part of this script in terms of future compatibility. Likewise, we
# need to set .iptables.localhostNodePorts=false so it won't try to
# set the sysctl associated with that. (The nftables mode never tries
# to set that sysctl.)
oc create -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
  namespace: kube-proxy-test
data:
  kube-proxy-config.yaml: |-
    apiVersion: kubeproxy.config.k8s.io/v1alpha1
    kind: KubeProxyConfiguration
    conntrack:
      maxPerCore: 0
      min: 0
      tcpCloseWaitTimeout: 0s
      tcpEstablishedTimeout: 0s
      udpStreamTimeout: 0s
      udpTimeout: 0s
    iptables:
      localhostNodePorts: false
    mode: ${PROXY_MODE}
EOF
echo "config is:"
oc get configmap -n kube-proxy-test config -o yaml | indent

# The --hostname-override is needed to fake out the node detection,
# since we aren't running in a host-network pod. (The fact that we're
# cheating here means we'll end up generating incorrect NodePort rules
# but that doesn't matter.)
oc create -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: kube-proxy
  namespace: kube-proxy-test
spec:
  containers:
  - name: kube-proxy
    image: ${KUBE_PROXY_IMAGE}
    command:
    - /bin/sh
    - -c
    - exec kube-proxy --hostname-override "\${NODENAME}" --config /config/kube-proxy-config.yaml -v 4
    env:
    - name: NODENAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
    securityContext:
      privileged: true
    volumeMounts:
    - mountPath: /config
      name: config
      readOnly: true
  serviceAccountName: kube-proxy-test
  volumes:
  - name: config
    configMap:
      name: config
EOF
echo "pod is:"
oc get pod -n kube-proxy-test kube-proxy -o yaml | indent
oc wait --for=condition=Ready -n kube-proxy-test pod/kube-proxy

echo "Waiting for kube-proxy to program initial ${PROXY_MODE} rules..."
function kube_proxy_synced() {
    oc exec -n kube-proxy-test kube-proxy -- curl -s http://127.0.0.1:10249/metrics > "${TMPDIR}/metrics.txt"
    grep -q '^kubeproxy_sync_proxy_rules_duration_seconds_count [^0]' "${TMPDIR}/metrics.txt"
}
synced=false
for count in $(seq 1 10); do
    date
    if kube_proxy_synced; then
        synced=true
        break
    fi
    sleep 5
done
date
if [[ "${synced}" != true ]]; then
    echo "kube-proxy failed to sync to ${PROXY_MODE}:"
    oc logs -n kube-proxy-test kube-proxy |& indent

    echo "last-seen metrics:"
    indent "${TMPDIR}/metrics.txt"

    exit 1
fi

# Dump the ruleset; since RHEL9 uses iptables-nft, kube-proxy's rules
# will show up in the nft ruleset regardless of whether kube-proxy is
# using iptables or nftables.
echo "Dumping rules"
oc exec -n kube-proxy-test kube-proxy -- nft list ruleset >& "${TMPDIR}/nft.out"

# We don't want to hardcode any assumptions about what kube-proxy's
# rules look like, but it necessarily must be the case that every
# clusterIP appears somewhere in the output. (We could look for
# endpoint IPs too, but that's more racy if there's any chance the
# cluster could be changing.)
exitcode=0
for service in kubernetes.default dns-default.openshift-dns router-default.openshift-ingress; do
    name="${service%.*}"
    namespace="${service#*.}"
    clusterIP="$(oc get service -n ${namespace} ${name} -o jsonpath='{.spec.clusterIP}')"
    echo "Looking for ${service} cluster IP (${clusterIP}) in ruleset"
    for ip in ${clusterIP}; do
        if ! grep --quiet --fixed-strings " ${ip} " "${TMPDIR}/nft.out"; then
            echo "Did not find IP ${ip} (from service ${name} in namespace ${namespace}) in ruleset" 1>&2
            exitcode=1
        fi
    done
done
echo ""

if [[ "${exitcode}" == 1 ]]; then
    echo "Ruleset was:"
    indent "${TMPDIR}/nft.out"

    echo "kube-proxy logs:"
    oc logs -n kube-proxy-test kube-proxy |& indent
fi

exit "${exitcode}"
