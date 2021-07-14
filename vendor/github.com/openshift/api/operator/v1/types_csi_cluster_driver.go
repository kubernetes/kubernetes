package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClusterCSIDriver is used to manage and configure CSI driver installed by default
// in OpenShift. An example configuration may look like:
//   apiVersion: operator.openshift.io/v1
//   kind: "ClusterCSIDriver"
//   metadata:
//     name: "ebs.csi.aws.com"
//   spec:
//     logLevel: Debug

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterCSIDriver object allows management and configuration of a CSI driver operator
// installed by default in OpenShift. Name of the object must be name of the CSI driver
// it operates. See CSIDriverName type for list of allowed values.
type ClusterCSIDriver struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ClusterCSIDriverSpec `json:"spec"`

	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ClusterCSIDriverStatus `json:"status"`
}

// CSIDriverName is the name of the CSI driver
type CSIDriverName string

// If you are adding a new driver name here, ensure that 0000_90_cluster_csi_driver_01_config.crd.yaml-merge-patch file is also updated with new driver name.
const (
	AWSEBSCSIDriver          CSIDriverName = "ebs.csi.aws.com"
	AWSEFSCSIDriver          CSIDriverName = "efs.csi.aws.com"
	AzureDiskCSIDriver       CSIDriverName = "disk.csi.azure.com"
	GCPPDCSIDriver           CSIDriverName = "pd.csi.storage.gke.io"
	CinderCSIDriver          CSIDriverName = "cinder.csi.openstack.org"
	VSphereCSIDriver         CSIDriverName = "csi.vsphere.vmware.com"
	ManilaCSIDriver          CSIDriverName = "manila.csi.openstack.org"
	OvirtCSIDriver           CSIDriverName = "csi.ovirt.org"
	KubevirtCSIDriver        CSIDriverName = "csi.kubevirt.io"
	SharedResourcesCSIDriver CSIDriverName = "csi.shared-resources.openshift.io"
)

// ClusterCSIDriverSpec is the desired behavior of CSI driver operator
type ClusterCSIDriverSpec struct {
	OperatorSpec `json:",inline"`
}

// ClusterCSIDriverStatus is the observed status of CSI driver operator
type ClusterCSIDriverStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// ClusterCSIDriverList contains a list of ClusterCSIDriver
type ClusterCSIDriverList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ClusterCSIDriver `json:"items"`
}
