# Tests for static Pods + CSI volume reconstruction

NOTE: these tests *require* the cluster under test to be Kubernetes In Docker
(kind)!

The tests here try to reconstruct CSI volumes at kubelet startup, without access
to the API server. Kind runs its API server as a static Pod, which is started
*after* kubelet finishes reconstruction. Here we use the API server as a guinea
pig that kubelet startup works correctly.
