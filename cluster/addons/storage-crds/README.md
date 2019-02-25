# Kubernetes CSI CRDs

The Kubernetes Container Storage Interface implementation defines some API objects as CRDs that Kubernetes components
including the Attach/Detach controller depend on.

If you are using CSI, it is recommended that you enable the relevant feature gates (e.g. `CSIDriverRegistry`, `CSINodeInfo`, etc.), and ensure the CRDs in this directory are installed.

These objects and their CRDs are defined in `staging/src/k8s.io/csi-api/pkg/crd/manifests`, the source of truth.
They are copied from that CRD manifest directory to this addon directory.
A unit test in `staging/src/k8s.io/csi-api/pkg/crd` verifies that this (and any other) copies of the manifest outside of `staging/src/k8s.io/csi-api/pkg/crd/manifests` do not drift from that source of truth.
If you need to make changes please make changes in the `staging/src/k8s.io/csi-api/pkg/crd/manifests` directory and then update this copy.

For more information, see: https://kubernetes-csi.github.io/docs/
