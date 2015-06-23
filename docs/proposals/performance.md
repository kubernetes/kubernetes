# Post-1.0 performance/scalability impovement ideas

This document is a collection of ideas for improving Kubernetes
performance/scalability post-1.0. Feel free to add to it. Please
link to relevant issues and assign an owner. Please only add things
that will actually improve performance.

## Performance/scalability goal

The next performance/scalability goal is 1000 nodes with 30 pods/node. We may wish to revisit the 30 pods/node part if we find that this is far above what real users need.

## Ideas

1. Change communication between apiserver and etcd to proto-based once v3 etcd is available [#8132](https://github.com/GoogleCloudPlatform/kubernetes/issues/8132) @wojtekt
2. Use BSON (http://bsonspec.org/) [#8132](https://github.com/GoogleCloudPlatform/kubernetes/issues/8132) @lavalamp
3. Eliminate "internal" representation and thus conversions [#8190](https://github.com/GoogleCloudPlatform/kubernetes/issues/8132) @briangrant
4. Alternative JSON parsers [#3338](https://github.com/GoogleCloudPlatform/kubernetes/issues/3338) @lavalamp
5. Watch improvements
   - implement watch at API server level [#8132](https://github.com/GoogleCloudPlatform/kubernetes/issues/8132) @wojtek-t
   - watch based on label query [#3295](https://github.com/GoogleCloudPlatform/kubernetes/issues/3295) @lavalamp
6. Change to proto for all cases, if necessary [#8132](https://github.com/GoogleCloudPlatform/kubernetes/issues/8132)
