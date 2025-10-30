This directory contains the Kubernetes E2E and integration tests. It is a
separate module to avoid tainting k8s.io/kubernetes with additional
dependencies which are only needed by those tests. Nothing from this directory
may be used in k8s.io/kubernetes.

See the "testutils" directory for helper code which may be used in
k8s.io/kubernetes and k8s.io/kubernetes/test.
