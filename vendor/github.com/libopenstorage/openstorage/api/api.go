package api

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/ptypes"
	"github.com/libopenstorage/openstorage/pkg/auth"
	"github.com/mohae/deepcopy"
)

// Strings for VolumeSpec
const (
	Name                     = "name"
	Token                    = "token"
	TokenSecret              = "token_secret"
	TokenSecretNamespace     = "token_secret_namespace"
	SpecNodes                = "nodes"
	SpecParent               = "parent"
	SpecEphemeral            = "ephemeral"
	SpecShared               = "shared"
	SpecJournal              = "journal"
	SpecSharedv4             = "sharedv4"
	SpecCascaded             = "cascaded"
	SpecSticky               = "sticky"
	SpecSecure               = "secure"
	SpecCompressed           = "compressed"
	SpecSize                 = "size"
	SpecScale                = "scale"
	SpecFilesystem           = "fs"
	SpecBlockSize            = "block_size"
	SpecQueueDepth           = "queue_depth"
	SpecHaLevel              = "repl"
	SpecPriority             = "io_priority"
	SpecSnapshotInterval     = "snap_interval"
	SpecSnapshotSchedule     = "snap_schedule"
	SpecAggregationLevel     = "aggregation_level"
	SpecDedupe               = "dedupe"
	SpecPassphrase           = "secret_key"
	SpecAutoAggregationValue = "auto"
	SpecGroup                = "group"
	SpecGroupEnforce         = "fg"
	SpecZones                = "zones"
	SpecRacks                = "racks"
	SpecRack                 = "rack"
	SpecRegions              = "regions"
	SpecLabels               = "labels"
	SpecPriorityAlias        = "priority_io"
	SpecIoProfile            = "io_profile"
	SpecAsyncIo              = "async_io"
	SpecEarlyAck             = "early_ack"
	SpecExportProtocol       = "export"
	SpecExportProtocolISCSI  = "iscsi"
	SpecExportProtocolPXD    = "pxd"
	SpecExportProtocolNFS    = "nfs"
	SpecExportProtocolCustom = "custom"
	SpecExportOptions        = "export_options"
	SpecExportOptionsEmpty   = "empty_export_options"
	SpecMountOptions         = "mount_options"
	// spec key cannot change due to parity with existing PSO storageclasses
	SpecFaCreateOptions      = "createoptions"
	SpecCSIMountOptions      = "csi_mount_options"
	SpecSharedv4MountOptions = "sharedv4_mount_options"
	SpecProxyProtocolS3      = "s3"
	SpecProxyProtocolPXD     = "pxd"
	SpecProxyProtocolNFS     = "nfs"
	SpecProxyEndpoint        = "proxy_endpoint"
	SpecProxyNFSSubPath      = "proxy_nfs_subpath"
	SpecProxyNFSExportPath   = "proxy_nfs_exportpath"
	SpecProxyS3Bucket        = "proxy_s3_bucket"
	// SpecBestEffortLocationProvisioning default is false. If set provisioning request will succeed
	// even if specified data location parameters could not be satisfied.
	SpecBestEffortLocationProvisioning = "best_effort_location_provisioning"
	// SpecForceUnsuppportedFsType is of type boolean and if true it sets
	// the VolumeSpec.force_unsupported_fs_type. When set to true it asks
	// the driver to use an unsupported value of VolumeSpec.format if possible
	SpecForceUnsupportedFsType = "force_unsupported_fs_type"
	// SpecMatchSrcVolProvision defaults to false. Applicable to cloudbackup restores only.
	// If set to "true", cloudbackup restore volume gets provisioned on same pools as
	// backup, allowing for inplace restore after.
	SpecMatchSrcVolProvision                = "match_src_vol_provision"
	SpecNodiscard                           = "nodiscard"
	StoragePolicy                           = "storagepolicy"
	SpecCowOnDemand                         = "cow_ondemand"
	SpecDirectIo                            = "direct_io"
	SpecScanPolicyTrigger                   = "scan_policy_trigger"
	SpecScanPolicyAction                    = "scan_policy_action"
	SpecProxyWrite                          = "proxy_write"
	SpecFastpath                            = "fastpath"
	SpecSharedv4ServiceType                 = "sharedv4_svc_type"
	SpecSharedv4ServiceName                 = "sharedv4_svc_name"
	SpecSharedv4FailoverStrategy            = "sharedv4_failover_strategy"
	SpecSharedv4FailoverStrategyNormal      = "normal"
	SpecSharedv4FailoverStrategyAggressive  = "aggressive"
	SpecSharedv4FailoverStrategyUnspecified = ""
	SpecSharedv4ExternalAccess              = "sharedv4_external_access"
	SpecAutoFstrim                          = "auto_fstrim"
	SpecBackendVolName                      = "pure_vol_name"
	SpecBackendType                         = "backend"
	SpecBackendPureBlock                    = "pure_block"
	SpecBackendPureFile                     = "pure_file"
	SpecPureFileExportRules                 = "pure_export_rules"
	SpecIoThrottleRdIOPS                    = "io_throttle_rd_iops"
	SpecIoThrottleWrIOPS                    = "io_throttle_wr_iops"
	SpecIoThrottleRdBW                      = "io_throttle_rd_bw"
	SpecIoThrottleWrBW                      = "io_throttle_wr_bw"
)

// OptionKey specifies a set of recognized query params.
const (
	// OptName query parameter used to lookup volume by name.
	OptName = "Name"
	// OptVolumeID query parameter used to lookup volume by ID.
	OptVolumeID = "VolumeID"
	// OptSnapID query parameter used to lookup snapshot by ID.
	OptSnapID = "SnapID"
	// OptLabel query parameter used to lookup volume by set of labels.
	OptLabel = "Label"
	// OptConfigLabel query parameter used to lookup volume by set of labels.
	OptConfigLabel = "ConfigLabel"
	// OptCumulative query parameter used to request cumulative stats.
	OptCumulative = "Cumulative"
	// OptTimeout query parameter used to indicate timeout seconds
	OptTimeoutSec = "TimeoutSec"
	// OptQuiesceID query parameter use for quiesce
	OptQuiesceID = "QuiesceID"
	// OptCredUUID is the UUID of the credential
	OptCredUUID = "CredUUID"
	// OptCredName indicates unique name of credential
	OptCredName = "CredName"
	// OptCredType  indicates type of credential
	OptCredType = "CredType"
	// OptCredEncrKey is the key used to encrypt data
	OptCredEncrKey = "CredEncrypt"
	// OptCredRegion indicates the region for s3
	OptCredRegion = "CredRegion"
	// OptCredDisableSSL indicated if SSL should be disabled
	OptCredDisableSSL = "CredDisableSSL"
	// OptCredDisablePathStyle does not enforce path style for s3
	OptCredDisablePathStyle = "CredDisablePathStyle"
	// OptCredStorageClass indicates the storage class to be used for puts
	// allowed values are STANDARD, STANDARD_IA,ONEZONE_IA, REDUCED_REDUNDANCY
	OptCredStorageClass = "CredStorageClass"
	// OptCredEndpoint indicate the cloud endpoint
	OptCredEndpoint = "CredEndpoint"
	// OptCredAccKey for s3
	OptCredAccessKey = "CredAccessKey"
	// OptCredSecretKey for s3
	OptCredSecretKey = "CredSecretKey"
	// OptCredBucket is the optional bucket name
	OptCredBucket = "CredBucket"
	// OptCredGoogleProjectID projectID for google cloud
	OptCredGoogleProjectID = "CredProjectID"
	// OptCredGoogleJsonKey for google cloud
	OptCredGoogleJsonKey = "CredJsonKey"
	// OptCredAzureAccountName is the account name for
	// azure as the cloud provider
	OptCredAzureAccountName = "CredAccountName"
	// OptOptCredAzureAccountKey is the accountkey for
	// azure as the cloud provider
	OptCredAzureAccountKey = "CredAccountKey"
	// Credential ownership key in params
	OptCredOwnership = "CredOwnership"
	// OptCredProxy proxy key in params
	OptCredProxy = "CredProxy"
	// OptCredNFSServer is the server address for NFS access
	OptCredNFSServer = "CredNFSServer"
	// OptCredNFSSubPath is the sub-path for objects
	OptCredNFSSubPath = "CredNFSSubPath"
	// OptCredNFSMountOpts is the optional mount options
	OptCredNFSMountOpts = "CredNFSMountOpts"
	// OptCredNFSTimeout is the optional timeout value
	OptCredNFSTimeoutSeconds = "CredNFSTimeout"
	// OptCredIAMPolicy if "true", indicates IAM creds to be used
	OptCredIAMPolicy = "CredIAMPolicy"
	// OptRemoteCredUUID is the UUID of the remote cluster credential
	OptRemoteCredUUID = "RemoteCredUUID"
	// OptCloudBackupID is the backID in the cloud
	OptCloudBackupID = "CloudBackID"
	// OptCloudBackupIgnoreCreds ignores credentials for incr backups
	OptCloudBackupIgnoreCreds = "CloudBackupIgnoreCreds"
	// OptSrcVolID is the source volume ID of the backup
	OptSrcVolID = "SrcVolID"
	// OptBkupOpState is the desired operational state
	// (stop/pause/resume) of backup/restore
	OptBkupOpState = "OpState"
	// OptBackupSchedUUID is the UUID of the backup-schedule
	OptBackupSchedUUID = "BkupSchedUUID"
	// OptVolumeSubFolder query parameter used to catalog a particular path inside a volume
	OptCatalogSubFolder = "subfolder"
	// OptCatalogMaxDepth query parameter used to limit the depth we return
	OptCatalogMaxDepth = "depth"
	// OptVolumeService query parameter used to request background volume services
	OptVolService = "volservice"
)

// Api clientserver Constants
const (
	OsdVolumePath        = "osd-volumes"
	OsdSnapshotPath      = "osd-snapshot"
	OsdCredsPath         = "osd-creds"
	OsdBackupPath        = "osd-backup"
	OsdMigratePath       = "osd-migrate"
	OsdMigrateStartPath  = OsdMigratePath + "/start"
	OsdMigrateCancelPath = OsdMigratePath + "/cancel"
	OsdMigrateStatusPath = OsdMigratePath + "/status"
	TimeLayout           = "Jan 2 15:04:05 UTC 2006"
)

const (
	// AutoAggregation value indicates driver to select aggregation level.
	AutoAggregation = math.MaxUint32
)

const (
	// gRPC root path used to extract service and API information
	SdkRootPath = "openstorage.api.OpenStorage"
)

// Node describes the state of a node.
// It includes the current physical state (CPU, memory, storage, network usage) as
// well as the containers running on the system.
//
// swagger:model
type Node struct {
	// Id of the node.
	Id string
	// SchedulerNodeName is name of the node in scheduler context. It can be
	// empty if unable to get the name from the scheduler.
	SchedulerNodeName string
	// Cpu usage of the node.
	Cpu float64 // percentage.
	// Total Memory of the node
	MemTotal uint64
	// Used Memory of the node
	MemUsed uint64
	// Free Memory of the node
	MemFree uint64
	// Average load (percentage)
	Avgload int
	// Node Status see (Status object)
	Status Status
	// GenNumber of the node
	GenNumber uint64
	// List of disks on this node.
	Disks map[string]StorageResource
	// List of storage pools this node supports
	Pools []StoragePool
	// Management IP
	MgmtIp string
	// Data IP
	DataIp string
	// Timestamp
	Timestamp time.Time
	// Start time of this node
	StartTime time.Time
	// Hostname of this node
	Hostname string
	// Node data for this node (EX: Public IP, Provider, City..)
	NodeData map[string]interface{}
	// User defined labels for node. Key Value pairs
	NodeLabels map[string]string
	// GossipPort is the port used by the gossip protocol
	GossipPort string
	// HWType is the type of the underlying hardware used by the node
	HWType HardwareType
	// Determine if the node is secure with authentication and authorization
	SecurityStatus StorageNode_SecurityStatus
	// SchedulerTopology topology information of the node in scheduler context
	SchedulerTopology *SchedulerTopology
}

// FluentDConfig describes ip and port of a fluentdhost.
// DEPRECATED
//
// swagger:model
type FluentDConfig struct {
	IP   string `json:"ip"`
	Port string `json:"port"`
}

// Cluster represents the state of the cluster.
//
// swagger:model
type Cluster struct {
	Status Status

	// Id of the cluster.
	//
	// required: true
	Id string

	// Id of the node on which this cluster object is initialized
	NodeId string

	// array of all the nodes in the cluster.
	Nodes []*Node

	// Management url for the cluster
	ManagementURL string

	// FluentD Host for the cluster
	FluentDConfig FluentDConfig
}

// CredCreateRequest is the input for CredCreate command
type CredCreateRequest struct {
	// InputParams is map describing cloud provide
	InputParams map[string]string
}

// CredCreateResponse is returned for CredCreate command
type CredCreateResponse struct {
	// UUID of the credential that was just created
	UUID string
}

// CredUpdateRequest is the input for CredsUpdate command
type CredUpdateRequest struct {
	// Name or the UUID of the credential being updated
	Name string
	// InputParams is map describing cloud provide
	InputParams map[string]string
}

// StatPoint represents the basic structure of a single Stat reported
// TODO: This is the first step to introduce stats in openstorage.
//       Follow up task is to introduce an API for logging stats
type StatPoint struct {
	// Name of the Stat
	Name string
	// Tags for the Stat
	Tags map[string]string
	// Fields and values of the stat
	Fields map[string]interface{}
	// Timestamp in Unix format
	Timestamp int64
}

type CloudBackupCreateRequest struct {
	// VolumeID of the volume for which cloudbackup is requested
	VolumeID string
	// CredentialUUID is cloud credential to be used for backup
	CredentialUUID string
	// Full indicates if full backup is desired even though incremental is possible
	Full bool
	// Name is optional unique id to be used for this backup
	// If not specified backup creates this by default
	Name string
	// Labels are list of key value pairs to tag the cloud backup. These labels
	// are stored in the metadata associated with the backup.
	Labels map[string]string
	// FullBackupFrequency indicates number of incremental backup after whcih
	// a fullbackup must be created. This is to override the default value for
	// manual/user triggerred backups and not applicable for scheduled backups.
	// Value of 0 retains the default behavior.
	FullBackupFrequency uint32
	// DeleteLocal indicates if local snap must be deleted after the
	// backup is complete
	DeleteLocal bool
}

type CloudBackupCreateResponse struct {
	// Name of the task performing this backup
	Name string
}

type CloudBackupGroupCreateRequest struct {
	// GroupID indicates backup request for a volumegroup with this group id
	GroupID string
	// Labels indicates backup request for a volume group with these labels
	Labels map[string]string
	// VolumeIDs are a list of volume IDs to use for the backup request
	// If multiple of GroupID, Labels or VolumeIDs are specified, volumes matching all of
	// them are backed up to cloud
	VolumeIDs []string
	// CredentialUUID is cloud credential to be used for backup
	CredentialUUID string
	// Full indicates if full backup is desired even though incremental is possible
	Full bool
	// DeleteLocal indicates if local snap must be deleted after the
	// backup is complete
	DeleteLocal bool
}

type CloudBackupRestoreRequest struct {
	// ID is the backup ID being restored
	ID string
	// RestoreVolumeName is optional volume Name of the new volume to be created
	// in the cluster for restoring the cloudbackup
	RestoreVolumeName string
	// CredentialUUID is the credential to be used for restore operation
	CredentialUUID string
	// NodeID is the optional NodeID for provisioning restore
	// volume (ResoreVolumeName should not be specified)
	NodeID string
	// Name is optional unique id to be used for this restore op
	// restore creates this by default
	Name string
	// Optional RestoreVolumeSpec allows some of the restoreVolume fields to be modified.
	// These fields default to the volume spec stored with cloudbackup.
	// The request fails if both RestoreVolSpec and NodeID are specified.
	Spec *RestoreVolumeSpec
	// Optional Locator for restoreVolume. Request fails if both Name and
	// locator are specified
	Locator *VolumeLocator
}

type CloudBackupGroupCreateResponse struct {
	// ID for this group of backups
	GroupCloudBackupID string
	// Names of the tasks performing this group backup
	Names []string
}

type CloudBackupRestoreResponse struct {
	// RestoreVolumeID is the volumeID to which the backup is being restored
	RestoreVolumeID string
	// Name of the task performing this restore
	Name string
}

type CloudBackupGenericRequest struct {
	// SrcVolumeID is optional Source VolumeID for the request
	SrcVolumeID string
	// ClusterID is the optional clusterID for the request
	ClusterID string
	// CredentialUUID is the credential for cloud to be used for the request
	CredentialUUID string
	// All if set to true, backups for all clusters in the cloud are processed
	All bool
	// StatusFilter indicates backups based on status
	StatusFilter CloudBackupStatusType
	// MetadataFilter indicates backups whose metadata has these kv pairs
	MetadataFilter map[string]string
	// CloudBackupID must be specified if one needs to enumerate known single
	// backup (format is clusteruuidORBucketName/srcVolId-SnapId(-incr). If
	// this is specified, everything else in the command is ignored
	CloudBackupID string
	// MissingSrcVol set to true enumerates cloudbackups for which srcVol is not
	// present in the cluster. Either the source volume is deleted or the
	// cloudbackup belongs to other cluster.( with older version this
	// information may be missing, and in such a case these will list as
	// missing cluster info field in enumeration). Specifying SrcVolumeID and
	// this flag at the same time is an error
	MissingSrcVolumes bool
}

type CloudBackupInfo struct {
	// ID is the ID of the cloud backup
	ID string
	// SrcVolumeID is Source volumeID of the backup
	SrcVolumeID string
	// SrcvolumeName is name of the sourceVolume of the backup
	SrcVolumeName string
	// Timestamp is the timestamp at which the source volume
	// was backed up to cloud
	Timestamp time.Time
	// Metadata associated with the backup
	Metadata map[string]string
	// Status indicates the status of the backup
	Status string
	// ClusterType indicates if the cloudbackup was uploaded by this
	// cluster. Could be unknown with older version cloudbackups
	ClusterType SdkCloudBackupClusterType_Value
	// Namespace to which this cloudbackup belongs to
	Namespace string
}

type CloudBackupEnumerateRequest struct {
	CloudBackupGenericRequest
	// MaxBackups indicates maxBackups to return in this enumerate list
	MaxBackups uint64
	// ContinuationToken returned in the enumerate response if all of the
	// requested backups could not be returned in one response
	ContinuationToken string
}

type CloudBackupEnumerateResponse struct {
	// Backups is list of backups in cloud for given volume/cluster/s
	Backups           []CloudBackupInfo
	ContinuationToken string
}

type CloudBackupDeleteRequest struct {
	// ID is the ID of the cloud backup
	ID string
	// CredentialUUID is the credential for cloud to be used for the request
	CredentialUUID string
	// Force Delete cloudbackup even if there are dependencies
	Force bool
}

type CloudBackupDeleteAllRequest struct {
	CloudBackupGenericRequest
}

type CloudBackupStatusRequest struct {
	// SrcVolumeID optional volumeID to list status of backup/restore
	SrcVolumeID string
	// Local indicates if only those backups/restores that are
	// active on current node must be returned
	Local bool
	// ID of the backup/restore task. If this is specified, SrcVolumeID is
	// ignored. This could be GroupCloudBackupId too, and in that case multiple
	// statuses belonging to the groupCloudBackupID is returned.
	ID string
}

type CloudBackupStatusRequestOld struct {
	// Old field for task ID
	Name string
	// New structure
	CloudBackupStatusRequest
}

type CloudBackupOpType string

const (
	CloudBackupOp  = CloudBackupOpType("Backup")
	CloudRestoreOp = CloudBackupOpType("Restore")
)

// Allowed storage classes s3
const (
	S3StorageClassStandard   = "STANDARD"
	S3StorageClassStandardIa = "STANDARD_IA"
)

type CloudBackupStatusType string

const (
	CloudBackupStatusNotStarted = CloudBackupStatusType("NotStarted")
	CloudBackupStatusDone       = CloudBackupStatusType("Done")
	CloudBackupStatusAborted    = CloudBackupStatusType("Aborted")
	CloudBackupStatusPaused     = CloudBackupStatusType("Paused")
	CloudBackupStatusStopped    = CloudBackupStatusType("Stopped")
	CloudBackupStatusActive     = CloudBackupStatusType("Active")
	CloudBackupStatusQueued     = CloudBackupStatusType("Queued")
	CloudBackupStatusFailed     = CloudBackupStatusType("Failed")
	// Invalid includes Failed, Stopped, and Aborted used as filter to enumerate
	// cloud backups
	CloudBackupStatusInvalid = CloudBackupStatusType("Invalid")
)

const (
	CloudBackupRequestedStatePause  = "pause"
	CloudBackupRequestedStateResume = "resume"
	CloudBackupRequestedStateStop   = "stop"
)

type CloudBackupStatus struct {
	// ID is the ID for the operation
	ID string
	// OpType indicates if this is a backup or restore
	OpType CloudBackupOpType
	// State indicates if the op is currently active/done/failed
	Status CloudBackupStatusType
	// BytesDone indicates Bytes uploaded/downloaded so far
	BytesDone uint64
	// BytesTotal is the total number of bytes being transferred
	BytesTotal uint64
	// EtaSeconds estimated time in seconds for backup/restore completion
	EtaSeconds int64
	// StartTime indicates Op's start time
	StartTime time.Time
	// CompletedTime indicates Op's completed time
	CompletedTime time.Time
	// NodeID is the ID of the node where this Op is active
	NodeID string
	// SrcVolumeID is either the volume being backed-up or target volume to
	// which a cloud backup is being restored
	SrcVolumeID string
	// Info currently indicates only failure cause in case of failed backup/restore
	Info []string
	// CredentialUUID used for this backup/restore op
	CredentialUUID string
	// GroupCloudBackupID is valid for backups that were started as part of group
	// cloudbackup request
	GroupCloudBackupID string
}

type CloudBackupStatusResponse struct {
	// statuses is list of currently active/failed/done backup/restores
	// map key is the id of the task
	Statuses map[string]CloudBackupStatus
}

type CloudBackupCatalogRequest struct {
	// ID is Backup ID in the cloud
	ID string
	// CredentialUUID is the credential for cloud
	CredentialUUID string
}

type CloudBackupCatalogResponse struct {
	// Contents is listing of backup contents
	Contents []string
}

type CloudBackupHistoryRequest struct {
	// SrcVolumeID is volumeID for which history of backup/restore
	// is being requested
	SrcVolumeID string
}

type CloudBackupHistoryItem struct {
	// SrcVolumeID is volume ID which was backedup
	SrcVolumeID string
	// TimeStamp is the time at which either backup completed/failed
	Timestamp time.Time
	// Status indicates whether backup was completed/failed
	Status string
}

type CloudBackupHistoryResponse struct {
	// HistoryList is list of past backup/restores in the cluster
	HistoryList []CloudBackupHistoryItem
}

type CloudBackupStateChangeRequest struct {
	// Name of the backup/restore task for which state change
	// is being requested
	Name string
	// RequestedState is desired state of the op
	// can be pause/resume/stop
	RequestedState string
}

type CloudBackupScheduleInfo struct {
	// SrcVolumeID is the schedule's source volume
	SrcVolumeID string
	// CredentialUUID is the cloud credential used with this schedule
	CredentialUUID string
	// Schedule is the frequence of backup
	Schedule string
	// MaxBackups are the maximum number of backups retained
	// in cloud.Older backups are deleted
	MaxBackups uint
	// GroupID indicates the group of volumes for this cloudbackup schedule
	GroupID string
	// Labels indicates a volume group for this cloudsnap schedule
	Labels map[string]string
	// Full indicates if scheduled backups must be full always
	Full bool
	// RetentionDays is the number of days that the scheduled backups will be kept
	// and after these number of days it will be deleted
	RetentionDays uint32
}

type CloudBackupSchedCreateRequest struct {
	CloudBackupScheduleInfo
}

// Callers must read the existing schedule and modify
// required fields
type CloudBackupSchedUpdateRequest struct {
	CloudBackupScheduleInfo
	// SchedUUID for which the schedule is being updated
	SchedUUID string
}

type CloudBackupGroupSchedCreateRequest struct {
	// GroupID indicates the group of volumes for which cloudbackup schedule is
	// being created
	GroupID string
	// Labels indicates a volume group for which this group cloudsnap schedule is
	// being created. If this is provided GroupId is not needed and vice-versa.
	Labels map[string]string
	// VolumeIDs are a list of volume IDs to use for the backup request
	// If multiple of GroupID, Labels or VolumeIDs are specified, volumes matching all of
	// them are backed up to cloud
	VolumeIDs []string
	// CredentialUUID is cloud credential to be used with this schedule
	CredentialUUID string
	// Schedule is the frequency of backup
	Schedule string
	// MaxBackups are the maximum number of backups retained
	// in cloud.Older backups are deleted
	MaxBackups uint
	// Full indicates if scheduled backups must be full always
	Full bool
	// RetentionDays is the number of days that the scheduled backups will be kept
	// and after these number of days it will be deleted
	RetentionDays uint32
}

type CloudBackupGroupSchedUpdateRequest struct {
	// Any parameters in this can be updated
	CloudBackupGroupSchedCreateRequest
	// UUID of the group schedule being upated
	SchedUUID string
}

type CloudBackupSchedCreateResponse struct {
	// UUID is the UUID of the newly created schedule
	UUID string
}

type CloudBackupSchedDeleteRequest struct {
	// UUID is UUID of the schedule to be deleted
	UUID string
}

type CloudBackupSchedEnumerateResponse struct {
	// Schedule is map of schedule uuid to scheduleInfo
	Schedules map[string]CloudBackupScheduleInfo
}

// Defines the response for CapacityUsage request
type CapacityUsageResponse struct {
	CapacityUsageInfo *CapacityUsageInfo
	// Describes the err if all of the usage details could not be obtained
	Error error
}

//
// DriverTypeSimpleValueOf returns the string format of DriverType
func DriverTypeSimpleValueOf(s string) (DriverType, error) {
	obj, err := simpleValueOf("driver_type", DriverType_value, s)
	return DriverType(obj), err
}

// SimpleString returns the string format of DriverType
func (x DriverType) SimpleString() string {
	return simpleString("driver_type", DriverType_name, int32(x))
}

// FSTypeSimpleValueOf returns the string format of FSType
func FSTypeSimpleValueOf(s string) (FSType, error) {
	obj, err := simpleValueOf("fs_type", FSType_value, s)
	return FSType(obj), err
}

// SimpleString returns the string format of DriverType
func (x FSType) SimpleString() string {
	return simpleString("fs_type", FSType_name, int32(x))
}

// CosTypeSimpleValueOf returns the string format of CosType
func CosTypeSimpleValueOf(s string) (CosType, error) {
	obj, exists := CosType_value[strings.ToUpper(s)]
	if !exists {
		return -1, fmt.Errorf("Invalid cos value: %s", s)
	}
	return CosType(obj), nil
}

// SimpleString returns the string format of CosType
func (x CosType) SimpleString() string {
	return simpleString("cos_type", CosType_name, int32(x))
}

// GraphDriverChangeTypeSimpleValueOf returns the string format of GraphDriverChangeType
func GraphDriverChangeTypeSimpleValueOf(s string) (GraphDriverChangeType, error) {
	obj, err := simpleValueOf("graph_driver_change_type", GraphDriverChangeType_value, s)
	return GraphDriverChangeType(obj), err
}

// SimpleString returns the string format of GraphDriverChangeType
func (x GraphDriverChangeType) SimpleString() string {
	return simpleString("graph_driver_change_type", GraphDriverChangeType_name, int32(x))
}

// VolumeActionParamSimpleValueOf returns the string format of VolumeAction
func VolumeActionParamSimpleValueOf(s string) (VolumeActionParam, error) {
	obj, err := simpleValueOf("volume_action_param", VolumeActionParam_value, s)
	return VolumeActionParam(obj), err
}

// SimpleString returns the string format of VolumeAction
func (x VolumeActionParam) SimpleString() string {
	return simpleString("volume_action_param", VolumeActionParam_name, int32(x))
}

// VolumeStateSimpleValueOf returns the string format of VolumeState
func VolumeStateSimpleValueOf(s string) (VolumeState, error) {
	obj, err := simpleValueOf("volume_state", VolumeState_value, s)
	return VolumeState(obj), err
}

// SimpleString returns the string format of VolumeState
func (x VolumeState) SimpleString() string {
	return simpleString("volume_state", VolumeState_name, int32(x))
}

// VolumeStatusSimpleValueOf returns the string format of VolumeStatus
func VolumeStatusSimpleValueOf(s string) (VolumeStatus, error) {
	obj, err := simpleValueOf("volume_status", VolumeStatus_value, s)
	return VolumeStatus(obj), err
}

// SimpleString returns the string format of VolumeStatus
func (x VolumeStatus) SimpleString() string {
	return simpleString("volume_status", VolumeStatus_name, int32(x))
}

// IoProfileSimpleValueOf returns the string format of IoProfile
func IoProfileSimpleValueOf(s string) (IoProfile, error) {
	obj, err := simpleValueOf("io_profile", IoProfile_value, s)
	return IoProfile(obj), err
}

// SimpleString returns the string format of IoProfile
func (x IoProfile) SimpleString() string {
	return simpleString("io_profile", IoProfile_name, int32(x))
}

// ProxyProtocolSimpleValueOf returns the string format of ProxyProtocol
func ProxyProtocolSimpleValueOf(s string) (ProxyProtocol, error) {
	obj, err := simpleValueOf("proxy_protocol", ProxyProtocol_value, s)
	return ProxyProtocol(obj), err
}

// SimpleString returns the string format of ProxyProtocol
func (x ProxyProtocol) SimpleString() string {
	return simpleString("proxy_protocol", ProxyProtocol_name, int32(x))
}

func simpleValueOf(typeString string, valueMap map[string]int32, s string) (int32, error) {
	obj, ok := valueMap[strings.ToUpper(fmt.Sprintf("%s_%s", typeString, s))]
	if !ok {
		return 0, fmt.Errorf("no openstorage.%s for %s", strings.ToUpper(typeString), s)
	}
	return obj, nil
}

func simpleString(typeString string, nameMap map[int32]string, v int32) string {
	s, ok := nameMap[v]
	if !ok {
		return strconv.Itoa(int(v))
	}
	return strings.TrimPrefix(strings.ToLower(s), fmt.Sprintf("%s_", strings.ToLower(typeString)))
}

// ScanPolicyTriggerValueof returns value of string
func ScanPolicy_ScanTriggerSimpleValueOf(s string) (ScanPolicy_ScanTrigger, error) {
	obj, err := simpleValueOf("scan_trigger", ScanPolicy_ScanTrigger_value, s)
	return ScanPolicy_ScanTrigger(obj), err
}

// SimpleString returns the string format of ScanPolicy_ScanTrigger
func (x ScanPolicy_ScanTrigger) SimpleString() string {
	return simpleString("scan_trigger", ScanPolicy_ScanTrigger_name, int32(x))
}

// ScanPolicyActioinValueof returns value of string
func ScanPolicy_ScanActionSimpleValueOf(s string) (ScanPolicy_ScanAction, error) {
	obj, err := simpleValueOf("scan_action", ScanPolicy_ScanAction_value, s)
	return ScanPolicy_ScanAction(obj), err
}

// SimpleString returns the string format of ScanPolicy_ScanAction
func (x ScanPolicy_ScanAction) SimpleString() string {
	return simpleString("scan_action", ScanPolicy_ScanAction_name, int32(x))
}

func toSec(ms uint64) uint64 {
	return ms / 1000
}

// WriteThroughput returns the write throughput
func (v *Stats) WriteThroughput() uint64 {
	intv := toSec(v.IntervalMs)
	if intv == 0 {
		return 0
	}
	return (v.WriteBytes) / intv
}

// ReadThroughput returns the read throughput
func (v *Stats) ReadThroughput() uint64 {
	intv := toSec(v.IntervalMs)
	if intv == 0 {
		return 0
	}
	return (v.ReadBytes) / intv
}

// Latency returns latency
func (v *Stats) Latency() uint64 {
	ops := v.Writes + v.Reads
	if ops == 0 {
		return 0
	}
	return (uint64)((v.IoMs * 1000) / ops)
}

// Read latency returns avg. time required for read operation to complete
func (v *Stats) ReadLatency() uint64 {
	if v.Reads == 0 {
		return 0
	}
	return (uint64)((v.ReadMs * 1000) / v.Reads)
}

// Write latency returns avg. time required for write operation to complete
func (v *Stats) WriteLatency() uint64 {
	if v.Writes == 0 {
		return 0
	}
	return (uint64)((v.WriteMs * 1000) / v.Writes)
}

// Iops returns iops
func (v *Stats) Iops() uint64 {
	intv := toSec(v.IntervalMs)
	if intv == 0 {
		return 0
	}
	return (v.Writes + v.Reads) / intv
}

// Scaled returns true if the volume is scaled.
func (v *Volume) Scaled() bool {
	return v.Spec.Scale > 1
}

// Contains returns true if locationConstraint is a member of volume's replication set.
func (m *Volume) Contains(locationConstraint string) bool {
	rsets := m.GetReplicaSets()
	for _, rset := range rsets {
		for _, node := range rset.Nodes {
			if node == locationConstraint {
				return true
			}
		}
	}

	// also check storage pool UUIDs
	for _, replSet := range m.ReplicaSets {
		for _, uid := range replSet.PoolUuids {
			if uid == locationConstraint {
				return true
			}
		}
	}

	return false
}

// Copy makes a deep copy of VolumeSpec
func (s *VolumeSpec) Copy() *VolumeSpec {
	spec := *s
	if s.ReplicaSet != nil {
		spec.ReplicaSet = &ReplicaSet{Nodes: make([]string, len(s.ReplicaSet.Nodes))}
		copy(spec.ReplicaSet.Nodes, s.ReplicaSet.Nodes)
	}
	return &spec
}

// Copy makes a deep copy of Node
func (s *Node) Copy() *Node {
	localCopy := deepcopy.Copy(*s)
	nodeCopy := localCopy.(Node)
	return &nodeCopy
}

func (v Volume) IsClone() bool {
	return v.Source != nil && len(v.Source.Parent) != 0 && !v.Readonly
}

func (v Volume) IsSnapshot() bool {
	return v.Source != nil && len(v.Source.Parent) != 0 && v.Readonly
}

func (v Volume) DisplayId() string {
	if v.Locator != nil {
		return fmt.Sprintf("%s (%s)", v.Locator.Name, v.Id)
	} else {
		return v.Id
	}
}

// ToStorageNode converts a Node structure to an exported gRPC StorageNode struct
func (s *Node) ToStorageNode() *StorageNode {
	node := &StorageNode{
		Id:                s.Id,
		SchedulerNodeName: s.SchedulerNodeName,
		Cpu:               s.Cpu,
		MemTotal:          s.MemTotal,
		MemUsed:           s.MemUsed,
		MemFree:           s.MemFree,
		AvgLoad:           int64(s.Avgload),
		Status:            s.Status,
		MgmtIp:            s.MgmtIp,
		DataIp:            s.DataIp,
		Hostname:          s.Hostname,
		HWType:            s.HWType,
		SecurityStatus:    s.SecurityStatus,
		SchedulerTopology: s.SchedulerTopology,
	}

	node.Disks = make(map[string]*StorageResource)
	for k, v := range s.Disks {
		// need to take the address of a local variable and not of v
		// since its address does not change
		vv := v
		node.Disks[k] = &vv
	}

	node.NodeLabels = make(map[string]string)
	for k, v := range s.NodeLabels {
		node.NodeLabels[k] = v
	}

	node.Pools = make([]*StoragePool, len(s.Pools))
	for i, v := range s.Pools {
		// need to take the address of a local variable and not of v
		// since its address does not change
		vv := v
		node.Pools[i] = &vv
	}

	return node
}

// ToStorageCluster converts a Cluster structure to an exported gRPC StorageCluster struct
func (c *Cluster) ToStorageCluster() *StorageCluster {
	cluster := &StorageCluster{
		Status: c.Status,

		// Due to history, the cluster ID is normally the name of the cluster, not the
		// unique identifier
		Name: c.Id,
	}

	return cluster
}

func CloudBackupStatusTypeToSdkCloudBackupStatusType(
	t CloudBackupStatusType,
) SdkCloudBackupStatusType {
	switch t {
	case CloudBackupStatusNotStarted:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeNotStarted
	case CloudBackupStatusDone:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeDone
	case CloudBackupStatusAborted:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeAborted
	case CloudBackupStatusPaused:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypePaused
	case CloudBackupStatusStopped:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeStopped
	case CloudBackupStatusActive:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeActive
	case CloudBackupStatusFailed:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeFailed
	case CloudBackupStatusQueued:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeQueued
	case CloudBackupStatusInvalid:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeInvalid
	default:
		return SdkCloudBackupStatusType_SdkCloudBackupStatusTypeUnknown
	}
}

func SdkCloudBackupStatusTypeToCloudBackupStatusString(
	t SdkCloudBackupStatusType,
) string {
	switch t {
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeNotStarted:
		return string(CloudBackupStatusNotStarted)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeDone:
		return string(CloudBackupStatusDone)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeAborted:
		return string(CloudBackupStatusAborted)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypePaused:
		return string(CloudBackupStatusPaused)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeStopped:
		return string(CloudBackupStatusStopped)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeActive:
		return string(CloudBackupStatusActive)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeFailed:
		return string(CloudBackupStatusFailed)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeQueued:
		return string(CloudBackupStatusQueued)
	case SdkCloudBackupStatusType_SdkCloudBackupStatusTypeInvalid:
		return string(CloudBackupStatusInvalid)
	default:
		return string(CloudBackupStatusFailed)
	}
}

func StringToSdkCloudBackupStatusType(s string) SdkCloudBackupStatusType {
	return CloudBackupStatusTypeToSdkCloudBackupStatusType(CloudBackupStatusType(s))
}

func (b *CloudBackupInfo) ToSdkCloudBackupInfo() *SdkCloudBackupInfo {
	info := &SdkCloudBackupInfo{
		Id:            b.ID,
		SrcVolumeId:   b.SrcVolumeID,
		SrcVolumeName: b.SrcVolumeName,
		Metadata:      b.Metadata,
		ClusterType:   b.ClusterType,
		Namespace:     b.Namespace,
	}

	info.Timestamp, _ = ptypes.TimestampProto(b.Timestamp)
	info.Status = StringToSdkCloudBackupStatusType(b.Status)

	return info
}

func (r *CloudBackupEnumerateResponse) ToSdkCloudBackupEnumerateWithFiltersResponse() *SdkCloudBackupEnumerateWithFiltersResponse {
	resp := &SdkCloudBackupEnumerateWithFiltersResponse{
		Backups: make([]*SdkCloudBackupInfo, len(r.Backups)),
	}

	for i, v := range r.Backups {
		resp.Backups[i] = v.ToSdkCloudBackupInfo()
	}
	resp.ContinuationToken = r.ContinuationToken
	return resp
}

func CloudBackupOpTypeToSdkCloudBackupOpType(t CloudBackupOpType) SdkCloudBackupOpType {
	switch t {
	case CloudBackupOp:
		return SdkCloudBackupOpType_SdkCloudBackupOpTypeBackupOp
	case CloudRestoreOp:
		return SdkCloudBackupOpType_SdkCloudBackupOpTypeRestoreOp
	default:
		return SdkCloudBackupOpType_SdkCloudBackupOpTypeUnknown
	}
}

func StringToSdkCloudBackupOpType(s string) SdkCloudBackupOpType {
	return CloudBackupOpTypeToSdkCloudBackupOpType(CloudBackupOpType(s))
}

func SdkCloudBackupOpTypeToCloudBackupOpType(t SdkCloudBackupOpType) CloudBackupOpType {
	switch t {
	case SdkCloudBackupOpType_SdkCloudBackupOpTypeBackupOp:
		return CloudBackupOp
	case SdkCloudBackupOpType_SdkCloudBackupOpTypeRestoreOp:
		return CloudRestoreOp
	default:
		return CloudBackupOpType("Unknown")
	}
}

func (s CloudBackupStatus) ToSdkCloudBackupStatus() *SdkCloudBackupStatus {
	status := &SdkCloudBackupStatus{
		BackupId:     s.ID,
		Optype:       CloudBackupOpTypeToSdkCloudBackupOpType(s.OpType),
		Status:       CloudBackupStatusTypeToSdkCloudBackupStatusType(s.Status),
		BytesDone:    s.BytesDone,
		NodeId:       s.NodeID,
		Info:         s.Info,
		CredentialId: s.CredentialUUID,
		SrcVolumeId:  s.SrcVolumeID,
		EtaSeconds:   s.EtaSeconds,
		BytesTotal:   s.BytesTotal,
	}

	status.StartTime, _ = ptypes.TimestampProto(s.StartTime)
	status.CompletedTime, _ = ptypes.TimestampProto(s.CompletedTime)

	return status
}

func (r *CloudBackupStatusResponse) ToSdkCloudBackupStatusResponse() *SdkCloudBackupStatusResponse {
	resp := &SdkCloudBackupStatusResponse{
		Statuses: make(map[string]*SdkCloudBackupStatus),
	}

	for k, v := range r.Statuses {
		resp.Statuses[k] = v.ToSdkCloudBackupStatus()
	}

	return resp
}

func (h CloudBackupHistoryItem) ToSdkCloudBackupHistoryItem() *SdkCloudBackupHistoryItem {
	item := &SdkCloudBackupHistoryItem{
		SrcVolumeId: h.SrcVolumeID,
		Status:      StringToSdkCloudBackupStatusType(h.Status),
	}

	item.Timestamp, _ = ptypes.TimestampProto(h.Timestamp)
	return item
}

func (r *CloudBackupHistoryResponse) ToSdkCloudBackupHistoryResponse() *SdkCloudBackupHistoryResponse {
	resp := &SdkCloudBackupHistoryResponse{
		HistoryList: make([]*SdkCloudBackupHistoryItem, len(r.HistoryList)),
	}

	for i, v := range r.HistoryList {
		resp.HistoryList[i] = v.ToSdkCloudBackupHistoryItem()
	}

	return resp
}

func (l *VolumeLocator) MergeVolumeSpecLabels(s *VolumeSpec) *VolumeLocator {
	if l.VolumeLabels == nil && len(s.GetVolumeLabels()) > 0 {
		l.VolumeLabels = make(map[string]string)
	}

	for k, v := range s.GetVolumeLabels() {
		l.VolumeLabels[k] = v
	}

	return l
}

func (v *Volume) IsPermitted(ctx context.Context, accessType Ownership_AccessType) bool {
	return v.GetSpec().IsPermitted(ctx, accessType)
}

func (v *VolumeSpec) IsPermitted(ctx context.Context, accessType Ownership_AccessType) bool {
	return v.GetOwnership().IsPermittedByContext(ctx, accessType)
}

func (v *VolumeSpec) IsPermittedFromUserInfo(user *auth.UserInfo, accessType Ownership_AccessType) bool {
	if v.IsPublic(accessType) {
		return true
	}

	if v.GetOwnership() != nil {
		return v.GetOwnership().IsPermitted(user, accessType)
	}
	return true
}

func (v *VolumeSpec) IsPublic(accessType Ownership_AccessType) bool {
	return v.GetOwnership() == nil || v.GetOwnership().IsPublic(accessType)
}

func (v *VolumeSpec) IsPureVolume() bool {
	return v.GetProxySpec() != nil && v.GetProxySpec().IsPureBackend()
}

// GetCloneCreatorOwnership returns the appropriate ownership for the
// new snapshot and if an update is required
func (v *VolumeSpec) GetCloneCreatorOwnership(ctx context.Context) (*Ownership, bool) {
	o := v.GetOwnership()

	// If there is user information, then auth is enabled
	if userinfo, ok := auth.NewUserInfoFromContext(ctx); ok {
		// Check if the owner is the one who cloned it
		if o != nil && o.IsOwner(userinfo) {
			return o, false
		}

		// Not the same owner, we now need new ownership.
		// This works for public volumes also.
		return OwnershipSetUsernameFromContext(ctx, nil), true
	}

	return o, false
}

// Check access permission of SdkStoragePolicy Objects

func (s *SdkStoragePolicy) IsPermitted(ctx context.Context, accessType Ownership_AccessType) bool {
	if s.IsPublic(accessType) {
		return true
	}

	// Storage Policy is not public, check permission
	if userinfo, ok := auth.NewUserInfoFromContext(ctx); ok {
		// Check Access
		return s.IsPermittedFromUserInfo(userinfo, accessType)
	} else {
		// There is no user information in the context so
		// authorization is not running
		return true
	}
}

func (s *SdkStoragePolicy) IsPermittedFromUserInfo(user *auth.UserInfo, accessType Ownership_AccessType) bool {
	if s.IsPublic(accessType) {
		return true
	}

	if s.GetOwnership() != nil {
		return s.GetOwnership().IsPermitted(user, accessType)
	}
	return true
}

func (s *SdkStoragePolicy) IsPublic(accessType Ownership_AccessType) bool {
	return s.GetOwnership() == nil || s.GetOwnership().IsPublic(accessType)
}

func CloudBackupRequestedStateToSdkCloudBackupRequestedState(
	t string,
) SdkCloudBackupRequestedState {
	switch t {
	case CloudBackupRequestedStateStop:
		return SdkCloudBackupRequestedState_SdkCloudBackupRequestedStateStop
	case CloudBackupRequestedStatePause:
		return SdkCloudBackupRequestedState_SdkCloudBackupRequestedStatePause
	case CloudBackupRequestedStateResume:
		return SdkCloudBackupRequestedState_SdkCloudBackupRequestedStateResume
	default:
		return SdkCloudBackupRequestedState_SdkCloudBackupRequestedStateUnknown
	}
}

// Helpers for volume state action
func (m *VolumeStateAction) IsAttach() bool {
	return m.GetAttach() == VolumeActionParam_VOLUME_ACTION_PARAM_ON
}

func (m *VolumeStateAction) IsDetach() bool {
	return m.GetAttach() == VolumeActionParam_VOLUME_ACTION_PARAM_OFF
}

func (m *VolumeStateAction) IsMount() bool {
	return m.GetMount() == VolumeActionParam_VOLUME_ACTION_PARAM_ON
}

func (m *VolumeStateAction) IsUnMount() bool {
	return m.GetMount() == VolumeActionParam_VOLUME_ACTION_PARAM_OFF
}

// IsAttached checks if a volume is attached
func (v *Volume) IsAttached() bool {
	return len(v.AttachedOn) > 0 &&
		v.State == VolumeState_VOLUME_STATE_ATTACHED &&
		v.AttachedState != AttachState_ATTACH_STATE_INTERNAL
}

// TokenSecretContext contains all nessesary information to get a
// token secret from any provider
type TokenSecretContext struct {
	SecretName      string
	SecretNamespace string
}

// ParseProxyEndpoint parses the proxy endpoint and returns the
// proxy protocol and the endpoint
func ParseProxyEndpoint(proxyEndpoint string) (ProxyProtocol, string) {
	if len(proxyEndpoint) == 0 {
		return ProxyProtocol_PROXY_PROTOCOL_INVALID, ""
	}
	tokens := strings.Split(proxyEndpoint, "://")
	if len(tokens) == 1 {
		return ProxyProtocol_PROXY_PROTOCOL_INVALID, tokens[0]
	} else if len(tokens) == 2 {
		switch tokens[0] {
		case SpecProxyProtocolS3:
			return ProxyProtocol_PROXY_PROTOCOL_S3, tokens[1]
		case SpecProxyProtocolNFS:
			return ProxyProtocol_PROXY_PROTOCOL_NFS, tokens[1]
		case SpecProxyProtocolPXD:
			return ProxyProtocol_PROXY_PROTOCOL_PXD, tokens[1]
		default:
			return ProxyProtocol_PROXY_PROTOCOL_INVALID, tokens[1]
		}
	}
	return ProxyProtocol_PROXY_PROTOCOL_INVALID, ""
}

func (s *ProxySpec) IsPureBackend() bool {
	return s.ProxyProtocol == ProxyProtocol_PROXY_PROTOCOL_PURE_BLOCK ||
		s.ProxyProtocol == ProxyProtocol_PROXY_PROTOCOL_PURE_FILE
}

func (s *ProxySpec) IsPureImport() bool {
	if !s.IsPureBackend() {
		return false
	}

	return (s.PureBlockSpec != nil && s.PureBlockSpec.FullVolName != "") || (s.PureFileSpec != nil && s.PureFileSpec.FullVolName != "")
}

func (s *ProxySpec) GetPureFullVolumeName() string {
	if !s.IsPureImport() {
		return ""
	}

	if s.PureBlockSpec != nil {
		return s.PureBlockSpec.FullVolName
	}

	if s.PureFileSpec != nil {
		return s.PureFileSpec.FullVolName
	}

	return ""
}
