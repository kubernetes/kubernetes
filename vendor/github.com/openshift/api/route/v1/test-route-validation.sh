#!/bin/bash

# This shell script runs a series of `oc` commands to create various OpenShift
# route objects, some invalid and some valid, and verifies that the API rejects
# the invalid ones and admits the valid ones.  Note that this script does not
# verify defaulting behavior and does not examine the rejection reason; it only
# checks whether the `oc create` command succeeds or fails.  This script
# requires a cluster and a kubeconfig in a location where oc will find it.

set -uo pipefail

expect_pass() {
  rc=$?
  if [[ $rc != 0 ]]
  then
    tput setaf 1
    echo "expected success: $*, got exit code $rc"
    tput sgr0
    exit 1
  fi
  tput setaf 2
  echo "got expected success: $*"
  tput sgr0
}

expect_fail() {
  rc=$?
  if [[ $rc = 0 ]]
  then
    tput setaf 1
    echo "expected failure: $*, got exit code $rc"
    exit 1
  fi
  tput setaf 2
  echo "got expected failure: $*"
  tput sgr0
}

delete_route() {
  oc -n openshift-ingress delete routes.route/testroute || exit 1
}

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  path: /
  tls:
    termination: passthrough
  to:
    kind: Service
    name: router-internal-default
EOF
expect_fail 'passthrough with nonempty path'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  path: /
  to:
    kind: Service
    name: router-internal-default
EOF
expect_pass 'non-TLS with nonempty path'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  path: /
  tls:
    termination: edge
  to:
    kind: Service
    name: router-internal-default
EOF
expect_pass 'edge-terminated with nonempty path'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  path: x
  tls:
    termination: edge
  to:
    kind: Service
    name: router-internal-default
EOF
expect_fail 'path starting with non-slash character'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  to:
    kind: Service
    name: router-internal-default
  wildcardPolicy: Subdomain
EOF
expect_fail 'spec.wildcardPolicy: Subdomain requires a nonempty value for spec.host'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  port:
    targetPort: ""
EOF
expect_fail 'cannot have empty spec.port.targetPort'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  port:
    targetPort: 0
EOF
expect_fail 'cannot have numeric 0 value for spec.port.targetPort'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  port:
    targetPort: "0" 
EOF
expect_pass 'can have string "0" value for spec.port.targetPort'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  port:
    targetPort: 1
EOF
expect_pass 'can have numeric 1 value for spec.port.targetPort'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  port:
    targetPort: x
EOF
expect_pass 'can have string "x" value for spec.port.targetPort'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  tls:
    termination: passthrough
  to:
    kind: Nonsense
    name: router-internal-default
EOF
expect_fail 'nonsense value for spec.to.kind'


oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  tls:
    termination: passthrough
  to:
    kind: Service
    name: ""
EOF
expect_fail 'spec.to.name cannot be empty'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
    weight: -1
EOF
expect_fail 'spec.to.weight cannot be negative'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
    weight: 300
EOF
expect_fail 'spec.to.weight cannot exceed 256'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
    weight: 100
EOF
expect_pass 'spec.to.weight has a valid value'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  alternateBackends:
  - name: router-internal-default
  - name: router-internal-default
  - name: router-internal-default
  - name: router-internal-default
EOF
expect_fail 'cannot have >3 values under spec.alternateBackends'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  alternateBackends:
  - name: router-internal-default
  - name: ""
  - name: router-internal-default
EOF
expect_fail 'cannot have empty spec.alternateBackends[*].name'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  alternateBackends:
  - name: router-internal-default
  - name: router-internal-default
  - name: router-internal-default
EOF
expect_pass 'valid spec.alternateBackends'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    certificate: "x"
EOF
expect_fail 'cannot have both spec.tls.termination: passthrough and nonempty spec.tls.certificate'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    key: "x"
EOF
expect_fail 'cannot have both spec.tls.termination: passthrough and nonempty spec.tls.key'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    caCertificate: "x"
EOF
expect_fail 'cannot have both spec.tls.termination: passthrough and nonempty spec.tls.caCertificate'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    destinationCACertificate: "x"
EOF
expect_fail 'cannot have both spec.tls.termination: passthrough and nonempty spec.tls.destinationCACertificate'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: edge
    destinationCACertificate: "x"
EOF
expect_fail 'cannot have both spec.tls.termination: edge and nonempty spec.tls.destinationCACertificate'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: nonsense
EOF
expect_fail 'cannot have nonsense value for spec.tls.insecureEdgeTerminationPolicy'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    insecureEdgeTerminationPolicy: Allow
EOF
expect_fail 'cannot have both spec.tls.termination: passthrough and spec.tls.insecureEdgeTerminationPolicy: Allow'

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    insecureEdgeTerminationPolicy: Redirect
EOF
expect_pass 'spec.tls.termination: passthrough is compatible with spec.tls.insecureEdgeTerminationPolicy: Redirect'
delete_route

oc create -f - <<'EOF'
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  namespace: openshift-ingress
  name: testroute
spec:
  host: test.foo
  to:
    name: router-internal-default
  tls:
    termination: passthrough
    insecureEdgeTerminationPolicy: None
EOF
expect_pass 'spec.tls.termination: passthrough is compatible with spec.tls.insecureEdgeTerminationPolicy: None'
delete_route
