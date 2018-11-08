# CSINodeInfo and CSIDriverInfo Usage and Lifecycle

CSINodeInfo is an API object representing CSI Information at the Node level.
CSINodeInfo contains a Spec and a Status, each containing Drivers represent the
driver spec and status respectively. The CSIDriverInfoSpec represents the
specification of the driver and is generally not changed, whereas the
CSIDriverInfoStatus represents the current status of the driver and can be
updated and reacted to by various components.

## Who creates it and when

CSINodeInfo is created by Kubelet when the first CSI Driver is installed to the
cluster and it is registered through the Kubelet device registration mechanism.

## Who updates CSIDriverInfo Spec and when

The CSIDriverInfoSpec for a driver is created upon installation of the CSI
Driver to the cluster and it is registered through the Kubelet device
registration mechanism. The spec is populated with information about the driver
through the nodeinfomanager (inside Kubelet) and will remain unchanged from then
on.

## Who updates Status and when

The CSIDriverInfoStatus for the driver is created upon installation of the CSI
Driver to the cluster (the same time as the spec) and it is registered through
the Kubelet device registration mechanism. The Status contains information about
installation and the required Volume Plugin Mechanism of the driver. When the
driver is installed/uninstalled through the Kubelet device registration
mechanism the Available flag is flipped from true/false respectively. The
migration status will also be updated when the flags for migration are set to
true/false on the Kubelet for that Driver on that node.

## Consumers of Status and Spec

Currently the only consumer of CSINodeInfo/CSIDriverInfo is the
csi-external-provisioner. In the future, the Attach Detach Controller (ADC) will
need to read this object to determine migration status on a per driver per node
basis. The creation of the CSINodeInfo object could possibly race with the
Attach/Detach controller as for CSI Migration the controller depend on the
existence of the API object for the driver but it will not have been created
yet. The ADC is expected to fail (and retry with exponential backoff) the
operation if it is expecting the object and it has not yet been created.

## Creation of CSINodeInfo object on Kubelet startup

For CSI Migration Alpha we expect any user who turns on the feature has both
Kubelet and ADC at a version where the CSINodeInfo's are being created on
Kubelet startup. We will not promote the feature to Beta (on by default) until
the CSINodeInfo's are being created on Kubelet startup for a 2 version skew to
prevent the case where the CSINodeInfo does not exist when the ADC depends on
it. This prevents the race described above becoming a permanent bad state.
