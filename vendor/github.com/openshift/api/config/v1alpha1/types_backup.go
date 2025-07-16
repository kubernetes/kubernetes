package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
//
// Backup provides configuration for performing backups of the openshift cluster.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=backups,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1482
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=AutomatedEtcdBackup
// +openshift:compatibility-gen:level=4
type Backup struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +required
	Spec BackupSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status BackupStatus `json:"status"`
}

type BackupSpec struct {
	// etcd specifies the configuration for periodic backups of the etcd cluster
	// +required
	EtcdBackupSpec EtcdBackupSpec `json:"etcd"`
}

type BackupStatus struct {
}

// EtcdBackupSpec provides configuration for automated etcd backups to the cluster-etcd-operator
type EtcdBackupSpec struct {

	// schedule defines the recurring backup schedule in Cron format
	// every 2 hours: 0 */2 * * *
	// every day at 3am: 0 3 * * *
	// Empty string means no opinion and the platform is left to choose a reasonable default which is subject to change without notice.
	// The current default is "no backups", but will change in the future.
	// +optional
	// +kubebuilder:validation:Pattern:=`^(@(annually|yearly|monthly|weekly|daily|hourly))|(\*|(?:\*|(?:[0-9]|(?:[1-5][0-9])))\/(?:[0-9]|(?:[1-5][0-9]))|(?:[0-9]|(?:[1-5][0-9]))(?:(?:\-[0-9]|\-(?:[1-5][0-9]))?|(?:\,(?:[0-9]|(?:[1-5][0-9])))*)) (\*|(?:\*|(?:\*|(?:[0-9]|1[0-9]|2[0-3])))\/(?:[0-9]|1[0-9]|2[0-3])|(?:[0-9]|1[0-9]|2[0-3])(?:(?:\-(?:[0-9]|1[0-9]|2[0-3]))?|(?:\,(?:[0-9]|1[0-9]|2[0-3]))*)) (\*|(?:[1-9]|(?:[12][0-9])|3[01])(?:(?:\-(?:[1-9]|(?:[12][0-9])|3[01]))?|(?:\,(?:[1-9]|(?:[12][0-9])|3[01]))*)) (\*|(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:(?:\-(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))?|(?:\,(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))*)) (\*|(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT)(?:(?:\-(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT))?|(?:\,(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT))*))$`
	Schedule string `json:"schedule"`

	// Cron Regex breakdown:
	// Allow macros: (@(annually|yearly|monthly|weekly|daily|hourly))
	// OR
	// Minute:
	//   (\*|(?:\*|(?:[0-9]|(?:[1-5][0-9])))\/(?:[0-9]|(?:[1-5][0-9]))|(?:[0-9]|(?:[1-5][0-9]))(?:(?:\-[0-9]|\-(?:[1-5][0-9]))?|(?:\,(?:[0-9]|(?:[1-5][0-9])))*))
	// Hour:
	//   (\*|(?:\*|(?:\*|(?:[0-9]|1[0-9]|2[0-3])))\/(?:[0-9]|1[0-9]|2[0-3])|(?:[0-9]|1[0-9]|2[0-3])(?:(?:\-(?:[0-9]|1[0-9]|2[0-3]))?|(?:\,(?:[0-9]|1[0-9]|2[0-3]))*))
	// Day of the Month:
	//   (\*|(?:[1-9]|(?:[12][0-9])|3[01])(?:(?:\-(?:[1-9]|(?:[12][0-9])|3[01]))?|(?:\,(?:[1-9]|(?:[12][0-9])|3[01]))*))
	// Month:
	//   (\*|(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:(?:\-(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))?|(?:\,(?:[1-9]|1[012]|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))*))
	// Day of Week:
	//   (\*|(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT)(?:(?:\-(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT))?|(?:\,(?:[0-6]|SUN|MON|TUE|WED|THU|FRI|SAT))*))
	//

	// The time zone name for the given schedule, see https://en.wikipedia.org/wiki/List_of_tz_database_time_zones.
	// If not specified, this will default to the time zone of the kube-controller-manager process.
	// See https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#time-zones
	// +optional
	// +kubebuilder:validation:Pattern:=`^([A-Za-z_]+([+-]*0)*|[A-Za-z_]+(\/[A-Za-z_]+){1,2})(\/GMT[+-]\d{1,2})?$`
	TimeZone string `json:"timeZone"`

	// Timezone regex breakdown:
	// ([A-Za-z_]+([+-]*0)*|[A-Za-z_]+(/[A-Za-z_]+){1,2}) - Matches either:
	//   [A-Za-z_]+([+-]*0)* - One or more alphabetical characters (uppercase or lowercase) or underscores, followed by a +0 or -0 to account for GMT+0 or GMT-0 (for the first part of the timezone identifier).
	//   [A-Za-z_]+(/[A-Za-z_]+){1,2} - One or more alphabetical characters (uppercase or lowercase) or underscores, followed by one or two occurrences of a forward slash followed by one or more alphabetical characters or underscores. This allows for matching timezone identifiers with 2 or 3 parts, e.g America/Argentina/Buenos_Aires
	// (/GMT[+-]\d{1,2})? - Makes the GMT offset suffix optional. It matches "/GMT" followed by either a plus ("+") or minus ("-") sign and one or two digits (the GMT offset)

	// retentionPolicy defines the retention policy for retaining and deleting existing backups.
	// +optional
	RetentionPolicy RetentionPolicy `json:"retentionPolicy"`

	// pvcName specifies the name of the PersistentVolumeClaim (PVC) which binds a PersistentVolume where the
	// etcd backup files would be saved
	// The PVC itself must always be created in the "openshift-etcd" namespace
	// If the PVC is left unspecified "" then the platform will choose a reasonable default location to save the backup.
	// In the future this would be backups saved across the control-plane master nodes.
	// +optional
	PVCName string `json:"pvcName"`
}

// RetentionType is the enumeration of valid retention policy types
// +enum
// +kubebuilder:validation:Enum:="RetentionNumber";"RetentionSize"
type RetentionType string

const (
	// RetentionTypeNumber sets the retention policy based on the number of backup files saved
	RetentionTypeNumber RetentionType = "RetentionNumber"
	// RetentionTypeSize sets the retention policy based on the total size of the backup files saved
	RetentionTypeSize RetentionType = "RetentionSize"
)

// RetentionPolicy defines the retention policy for retaining and deleting existing backups.
// This struct is a discriminated union that allows users to select the type of retention policy from the supported types.
// +union
type RetentionPolicy struct {
	// retentionType sets the type of retention policy.
	// Currently, the only valid policies are retention by number of backups (RetentionNumber), by the size of backups (RetentionSize). More policies or types may be added in the future.
	// Empty string means no opinion and the platform is left to choose a reasonable default which is subject to change without notice.
	// The current default is RetentionNumber with 15 backups kept.
	// +unionDiscriminator
	// +required
	// +kubebuilder:validation:Enum:="";"RetentionNumber";"RetentionSize"
	RetentionType RetentionType `json:"retentionType"`

	// retentionNumber configures the retention policy based on the number of backups
	// +optional
	RetentionNumber *RetentionNumberConfig `json:"retentionNumber,omitempty"`

	// retentionSize configures the retention policy based on the size of backups
	// +optional
	RetentionSize *RetentionSizeConfig `json:"retentionSize,omitempty"`
}

// RetentionNumberConfig specifies the configuration of the retention policy on the number of backups
type RetentionNumberConfig struct {
	// maxNumberOfBackups defines the maximum number of backups to retain.
	// If the existing number of backups saved is equal to MaxNumberOfBackups then
	// the oldest backup will be removed before a new backup is initiated.
	// +kubebuilder:validation:Minimum=1
	// +required
	MaxNumberOfBackups int `json:"maxNumberOfBackups"`
}

// RetentionSizeConfig specifies the configuration of the retention policy on the total size of backups
type RetentionSizeConfig struct {
	// maxSizeOfBackupsGb defines the total size in GB of backups to retain.
	// If the current total size backups exceeds MaxSizeOfBackupsGb then
	// the oldest backup will be removed before a new backup is initiated.
	// +kubebuilder:validation:Minimum=1
	// +required
	MaxSizeOfBackupsGb int `json:"maxSizeOfBackupsGb"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BackupList is a collection of items
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type BackupList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`
	Items           []Backup `json:"items"`
}
