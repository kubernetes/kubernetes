package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +kubebuilder:resource:scope=Cluster
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
//
// # EtcdBackup provides configuration options and status for a one-time backup attempt of the etcd cluster
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type EtcdBackup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec EtcdBackupSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +kubebuilder:validation:Optional
	// +optional
	Status EtcdBackupStatus `json:"status"`
}

type EtcdBackupSpec struct {
	// PVCName specifies the name of the PersistentVolumeClaim (PVC) which binds a PersistentVolume where the
	// etcd backup file would be saved
	// The PVC itself must always be created in the "openshift-etcd" namespace
	// If the PVC is left unspecified "" then the platform will choose a reasonable default location to save the backup.
	// In the future this would be backups saved across the control-plane master nodes.
	// +kubebuilder:validation:Optional
	// +optional
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="pvcName is immutable once set"
	PVCName string `json:"pvcName"`
}

// +kubebuilder:validation:Optional
type EtcdBackupStatus struct {
	// conditions provide details on the status of the etcd backup job.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions" patchStrategy:"merge" patchMergeKey:"type"`

	// backupJob is the reference to the Job that executes the backup.
	// Optional
	// +kubebuilder:validation:Optional
	BackupJob *BackupJobReference `json:"backupJob"`
}

// BackupJobReference holds a reference to the batch/v1 Job created to run the etcd backup
type BackupJobReference struct {

	// namespace is the namespace of the Job.
	// this is always expected to be "openshift-etcd" since the user provided PVC
	// is also required to be in "openshift-etcd"
	// Required
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern:=`^openshift-etcd$`
	Namespace string `json:"namespace"`

	// name is the name of the Job.
	// Required
	// +kubebuilder:validation:Required
	Name string `json:"name"`
}

type BackupConditionReason string

var (
	// BackupPending is added to the EtcdBackupStatus Conditions when the etcd backup is pending.
	BackupPending BackupConditionReason = "BackupPending"

	// BackupCompleted is added to the EtcdBackupStatus Conditions when the etcd backup has completed.
	BackupCompleted BackupConditionReason = "BackupCompleted"

	// BackupFailed is added to the EtcdBackupStatus Conditions when the etcd backup has failed.
	BackupFailed BackupConditionReason = "BackupFailed"

	// BackupSkipped is added to the EtcdBackupStatus Conditions when the etcd backup has been skipped.
	BackupSkipped BackupConditionReason = "BackupSkipped"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EtcdBackupList is a collection of items
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type EtcdBackupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []EtcdBackup `json:"items"`
}
