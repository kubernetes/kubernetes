package csiinlinevolumesecurity

// The CSIInlineVolumeSecurity admission plugin inspects inline volumes
// on pod creation and compares the security.openshift.io/csi-ephemeral-volume-profile
// label on the associated CSIDriver object to the pod security profile on the namespace.
// Admission is only allowed if the namespace enforces a profile of equal or greater
// permission compared to the profile label for the CSIDriver.
