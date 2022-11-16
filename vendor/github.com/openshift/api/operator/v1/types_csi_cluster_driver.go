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
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
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

// +kubebuilder:validation:Enum="";Managed;Unmanaged;Removed
// StorageClassStateName defines various configuration states for storageclass management
// and reconciliation by CSI operator.
type StorageClassStateName string

const (
	// ManagedStorageClass means that the operator is actively managing its storage classes.
	// Most manual changes made by cluster admin to storageclass will be wiped away by CSI
	// operator if StorageClassState is set to Managed.
	ManagedStorageClass StorageClassStateName = "Managed"
	// UnmanagedStorageClass means that the operator is not actively managing storage classes.
	// If StorageClassState is Unmanaged then CSI operator will not be actively reconciling storage class
	// it previously created. This can be useful if cluster admin wants to modify storage class installed
	// by CSI operator.
	UnmanagedStorageClass StorageClassStateName = "Unmanaged"
	// RemovedStorageClass instructs the operator to remove the storage class.
	// If StorageClassState is Removed - CSI operator will delete storage classes it created
	// previously. This can be useful in clusters where cluster admins want to prevent
	// creation of dynamically provisioned volumes but still need rest of the features
	// provided by CSI operator and driver.
	RemovedStorageClass StorageClassStateName = "Removed"
)

// If you are adding a new driver name here, ensure that 0000_90_cluster_csi_driver_01_config.crd.yaml-merge-patch file is also updated with new driver name.
const (
	AWSEBSCSIDriver          CSIDriverName = "ebs.csi.aws.com"
	AWSEFSCSIDriver          CSIDriverName = "efs.csi.aws.com"
	AzureDiskCSIDriver       CSIDriverName = "disk.csi.azure.com"
	AzureFileCSIDriver       CSIDriverName = "file.csi.azure.com"
	GCPFilestoreCSIDriver    CSIDriverName = "filestore.csi.storage.gke.io"
	GCPPDCSIDriver           CSIDriverName = "pd.csi.storage.gke.io"
	CinderCSIDriver          CSIDriverName = "cinder.csi.openstack.org"
	VSphereCSIDriver         CSIDriverName = "csi.vsphere.vmware.com"
	ManilaCSIDriver          CSIDriverName = "manila.csi.openstack.org"
	OvirtCSIDriver           CSIDriverName = "csi.ovirt.org"
	KubevirtCSIDriver        CSIDriverName = "csi.kubevirt.io"
	SharedResourcesCSIDriver CSIDriverName = "csi.sharedresource.openshift.io"
	AlibabaDiskCSIDriver     CSIDriverName = "diskplugin.csi.alibabacloud.com"
	IBMVPCBlockCSIDriver     CSIDriverName = "vpc.block.csi.ibm.io"
	IBMPowerVSBlockCSIDriver CSIDriverName = "powervs.csi.ibm.com"
)

// ClusterCSIDriverSpec is the desired behavior of CSI driver operator
type ClusterCSIDriverSpec struct {
	OperatorSpec `json:",inline"`
	// StorageClassState determines if CSI operator should create and manage storage classes.
	// If this field value is empty or Managed - CSI operator will continuously reconcile
	// storage class and create if necessary.
	// If this field value is Unmanaged - CSI operator will not reconcile any previously created
	// storage class.
	// If this field value is Removed - CSI operator will delete the storage class it created previously.
	// When omitted, this means the user has no opinion and the platform chooses a reasonable default,
	// which is subject to change over time.
	// The current default behaviour is Managed.
	// +optional
	StorageClassState StorageClassStateName `json:"storageClassState,omitempty"`

	// driverConfig can be used to specify platform specific driver configuration.
	// When omitted, this means no opinion and the platform is left to choose reasonable
	// defaults. These defaults are subject to change over time.
	// +optional
	DriverConfig CSIDriverConfigSpec `json:"driverConfig"`
}

// CSIDriverType indicates type of CSI driver being configured.
// +kubebuilder:validation:Enum="";vSphere
type CSIDriverType string

const (
	VSphereDriverType CSIDriverType = "vSphere"
)

// CSIDriverConfigSpec defines configuration spec that can be
// used to optionally configure a specific CSI Driver.
// +union
type CSIDriverConfigSpec struct {
	// driverType indicates type of CSI driver for which the
	// driverConfig is being applied to.
	//
	// Valid values are:
	//
	// * vSphere
	//
	// Allows configuration of vsphere CSI driver topology.
	//
	// ---
	// Consumers should treat unknown values as a NO-OP.
	//
	// +kubebuilder:validation:Required
	// +unionDiscriminator
	DriverType CSIDriverType `json:"driverType"`

	// vsphere is used to configure the vsphere CSI driver.
	// +optional
	VSphere *VSphereCSIDriverConfigSpec `json:"vSphere,omitempty"`
}

// VSphereCSIDriverConfigSpec defines properties that
// can be configured for vsphere CSI driver.
type VSphereCSIDriverConfigSpec struct {
	// topologyCategories indicates tag categories with which
	// vcenter resources such as hostcluster or datacenter were tagged with.
	// If cluster Infrastructure object has a topology, values specified in
	// Infrastructure object will be used and modifications to topologyCategories
	// will be rejected.
	// +optional
	TopologyCategories []string `json:"topologyCategories,omitempty"`
}

// ClusterCSIDriverStatus is the observed status of CSI driver operator
type ClusterCSIDriverStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// ClusterCSIDriverList contains a list of ClusterCSIDriver
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ClusterCSIDriverList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ClusterCSIDriver `json:"items"`
}
