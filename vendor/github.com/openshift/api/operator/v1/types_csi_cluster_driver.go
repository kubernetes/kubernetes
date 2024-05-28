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
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=clustercsidrivers,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/701
// +openshift:file-pattern=cvoRunLevel=0000_90,operatorName=csi-driver,operatorOrdering=01

// ClusterCSIDriver object allows management and configuration of a CSI driver operator
// installed by default in OpenShift. Name of the object must be name of the CSI driver
// it operates. See CSIDriverName type for list of allowed values.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ClusterCSIDriver struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
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
	SecretsStoreCSIDriver    CSIDriverName = "secrets-store.csi.k8s.io"
	SambaCSIDriver           CSIDriverName = "smb.csi.k8s.io"
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
// +kubebuilder:validation:Enum="";AWS;Azure;GCP;IBMCloud;vSphere
type CSIDriverType string

const (
	AWSDriverType      CSIDriverType = "AWS"
	AzureDriverType    CSIDriverType = "Azure"
	GCPDriverType      CSIDriverType = "GCP"
	IBMCloudDriverType CSIDriverType = "IBMCloud"
	VSphereDriverType  CSIDriverType = "vSphere"
)

// CSIDriverConfigSpec defines configuration spec that can be
// used to optionally configure a specific CSI Driver.
// +kubebuilder:validation:XValidation:rule="has(self.driverType) && self.driverType == 'IBMCloud' ? has(self.ibmcloud) : !has(self.ibmcloud)",message="ibmcloud must be set if driverType is 'IBMCloud', but remain unset otherwise"
// +union
type CSIDriverConfigSpec struct {
	// driverType indicates type of CSI driver for which the
	// driverConfig is being applied to.
	// Valid values are: AWS, Azure, GCP, IBMCloud, vSphere and omitted.
	// Consumers should treat unknown values as a NO-OP.
	// +kubebuilder:validation:Required
	// +unionDiscriminator
	DriverType CSIDriverType `json:"driverType"`

	// aws is used to configure the AWS CSI driver.
	// +optional
	AWS *AWSCSIDriverConfigSpec `json:"aws,omitempty"`

	// azure is used to configure the Azure CSI driver.
	// +optional
	Azure *AzureCSIDriverConfigSpec `json:"azure,omitempty"`

	// gcp is used to configure the GCP CSI driver.
	// +optional
	GCP *GCPCSIDriverConfigSpec `json:"gcp,omitempty"`

	// ibmcloud is used to configure the IBM Cloud CSI driver.
	// +optional
	IBMCloud *IBMCloudCSIDriverConfigSpec `json:"ibmcloud,omitempty"`

	// vsphere is used to configure the vsphere CSI driver.
	// +optional
	VSphere *VSphereCSIDriverConfigSpec `json:"vSphere,omitempty"`
}

// AWSCSIDriverConfigSpec defines properties that can be configured for the AWS CSI driver.
type AWSCSIDriverConfigSpec struct {
	// kmsKeyARN sets the cluster default storage class to encrypt volumes with a user-defined KMS key,
	// rather than the default KMS key used by AWS.
	// The value may be either the ARN or Alias ARN of a KMS key.
	// +kubebuilder:validation:Pattern:=`^arn:(aws|aws-cn|aws-us-gov|aws-iso|aws-iso-b|aws-iso-e|aws-iso-f):kms:[a-z0-9-]+:[0-9]{12}:(key|alias)\/.*$`
	// +optional
	KMSKeyARN string `json:"kmsKeyARN,omitempty"`
}

// AzureDiskEncryptionSet defines the configuration for a disk encryption set.
type AzureDiskEncryptionSet struct {
	// subscriptionID defines the Azure subscription that contains the disk encryption set.
	// The value should meet the following conditions:
	// 1. It should be a 128-bit number.
	// 2. It should be 36 characters (32 hexadecimal characters and 4 hyphens) long.
	// 3. It should be displayed in five groups separated by hyphens (-).
	// 4. The first group should be 8 characters long.
	// 5. The second, third, and fourth groups should be 4 characters long.
	// 6. The fifth group should be 12 characters long.
	// An Example SubscrionID: f2007bbf-f802-4a47-9336-cf7c6b89b378
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength:=36
	// +kubebuilder:validation:Pattern:=`^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$`
	SubscriptionID string `json:"subscriptionID"`

	// resourceGroup defines the Azure resource group that contains the disk encryption set.
	// The value should consist of only alphanumberic characters,
	// underscores (_), parentheses, hyphens and periods.
	// The value should not end in a period and be at most 90 characters in
	// length.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength:=90
	// +kubebuilder:validation:Pattern:=`^[\w\.\-\(\)]*[\w\-\(\)]$`
	ResourceGroup string `json:"resourceGroup"`

	// name is the name of the disk encryption set that will be set on the default storage class.
	// The value should consist of only alphanumberic characters,
	// underscores (_), hyphens, and be at most 80 characters in length.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength:=80
	// +kubebuilder:validation:Pattern:=`^[a-zA-Z0-9\_-]+$`
	Name string `json:"name"`
}

// AzureCSIDriverConfigSpec defines properties that can be configured for the Azure CSI driver.
type AzureCSIDriverConfigSpec struct {
	// diskEncryptionSet sets the cluster default storage class to encrypt volumes with a
	// customer-managed encryption set, rather than the default platform-managed keys.
	// +optional
	DiskEncryptionSet *AzureDiskEncryptionSet `json:"diskEncryptionSet,omitempty"`
}

// GCPKMSKeyReference gathers required fields for looking up a GCP KMS Key
type GCPKMSKeyReference struct {
	// name is the name of the customer-managed encryption key to be used for disk encryption.
	// The value should correspond to an existing KMS key and should
	// consist of only alphanumeric characters, hyphens (-) and underscores (_),
	// and be at most 63 characters in length.
	// +kubebuilder:validation:Pattern:=`^[a-zA-Z0-9\_-]+$`
	// +kubebuilder:validation:MinLength:=1
	// +kubebuilder:validation:MaxLength:=63
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// keyRing is the name of the KMS Key Ring which the KMS Key belongs to.
	// The value should correspond to an existing KMS key ring and should
	// consist of only alphanumeric characters, hyphens (-) and underscores (_),
	// and be at most 63 characters in length.
	// +kubebuilder:validation:Pattern:=`^[a-zA-Z0-9\_-]+$`
	// +kubebuilder:validation:MinLength:=1
	// +kubebuilder:validation:MaxLength:=63
	// +kubebuilder:validation:Required
	KeyRing string `json:"keyRing"`

	// projectID is the ID of the Project in which the KMS Key Ring exists.
	// It must be 6 to 30 lowercase letters, digits, or hyphens.
	// It must start with a letter. Trailing hyphens are prohibited.
	// +kubebuilder:validation:Pattern:=`^[a-z][a-z0-9-]+[a-z0-9]$`
	// +kubebuilder:validation:MinLength:=6
	// +kubebuilder:validation:MaxLength:=30
	// +kubebuilder:validation:Required
	ProjectID string `json:"projectID"`

	// location is the GCP location in which the Key Ring exists.
	// The value must match an existing GCP location, or "global".
	// Defaults to global, if not set.
	// +kubebuilder:validation:Pattern:=`^[a-zA-Z0-9\_-]+$`
	// +optional
	Location string `json:"location,omitempty"`
}

// GCPCSIDriverConfigSpec defines properties that can be configured for the GCP CSI driver.
type GCPCSIDriverConfigSpec struct {
	// kmsKey sets the cluster default storage class to encrypt volumes with customer-supplied
	// encryption keys, rather than the default keys managed by GCP.
	// +optional
	KMSKey *GCPKMSKeyReference `json:"kmsKey,omitempty"`
}

// IBMCloudCSIDriverConfigSpec defines the properties that can be configured for the IBM Cloud CSI driver.
type IBMCloudCSIDriverConfigSpec struct {
	// encryptionKeyCRN is the IBM Cloud CRN of the customer-managed root key to use
	// for disk encryption of volumes for the default storage classes.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength:=154
	// +kubebuilder:validation:MinLength:=144
	// +kubebuilder:validation:Pattern:=`^crn:v[0-9]+:bluemix:(public|private):(kms|hs-crypto):[a-z-]+:a/[0-9a-f]+:[0-9a-f-]{36}:key:[0-9a-f-]{36}$`
	EncryptionKeyCRN string `json:"encryptionKeyCRN"`
}

// VSphereCSIDriverConfigSpec defines properties that
// can be configured for vsphere CSI driver.
type VSphereCSIDriverConfigSpec struct {
	// topologyCategories indicates tag categories with which
	// vcenter resources such as hostcluster or datacenter were tagged with.
	// If cluster Infrastructure object has a topology, values specified in
	// Infrastructure object will be used and modifications to topologyCategories
	// will be rejected.
	// +listType=atomic
	// +optional
	TopologyCategories []string `json:"topologyCategories,omitempty"`

	// globalMaxSnapshotsPerBlockVolume is a global configuration parameter that applies to volumes on all kinds of
	// datastores. If omitted, the platform chooses a default, which is subject to change over time, currently that default is 3.
	// Snapshots can not be disabled using this parameter.
	// Increasing number of snapshots above 3 can have negative impact on performance, for more details see: https://kb.vmware.com/s/article/1025279
	// Volume snapshot documentation: https://docs.vmware.com/en/VMware-vSphere-Container-Storage-Plug-in/3.0/vmware-vsphere-csp-getting-started/GUID-E0B41C69-7EEB-450F-A73D-5FD2FF39E891.html
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=32
	// +openshift:enable:FeatureGate=VSphereDriverConfiguration
	// +optional
	GlobalMaxSnapshotsPerBlockVolume *uint32 `json:"globalMaxSnapshotsPerBlockVolume,omitempty"`

	// granularMaxSnapshotsPerBlockVolumeInVSAN is a granular configuration parameter on vSAN datastore only. It
	// overrides GlobalMaxSnapshotsPerBlockVolume if set, while it falls back to the global constraint if unset.
	// Snapshots for VSAN can not be disabled using this parameter.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=32
	// +openshift:enable:FeatureGate=VSphereDriverConfiguration
	// +optional
	GranularMaxSnapshotsPerBlockVolumeInVSAN *uint32 `json:"granularMaxSnapshotsPerBlockVolumeInVSAN,omitempty"`

	// granularMaxSnapshotsPerBlockVolumeInVVOL is a granular configuration parameter on Virtual Volumes datastore only.
	// It overrides GlobalMaxSnapshotsPerBlockVolume if set, while it falls back to the global constraint if unset.
	// Snapshots for VVOL can not be disabled using this parameter.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=32
	// +openshift:enable:FeatureGate=VSphereDriverConfiguration
	// +optional
	GranularMaxSnapshotsPerBlockVolumeInVVOL *uint32 `json:"granularMaxSnapshotsPerBlockVolumeInVVOL,omitempty"`
}

// ClusterCSIDriverStatus is the observed status of CSI driver operator
type ClusterCSIDriverStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterCSIDriverList contains a list of ClusterCSIDriver
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ClusterCSIDriverList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []ClusterCSIDriver `json:"items"`
}
