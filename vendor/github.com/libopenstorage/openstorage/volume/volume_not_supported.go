package volume

import (
	"context"

	"github.com/libopenstorage/openstorage/api"
)

var (
	// BlockNotSupported is a default (null) block driver implementation.  This can be
	// used by drivers that do not want to (or care about) implementing the attach,
	// format and detach interfaces.
	BlockNotSupported = &blockNotSupported{}
	// SnapshotNotSupported is a null snapshot driver implementation. This can be used
	// by drivers that do not want to implement the snapshot interface
	SnapshotNotSupported = &snapshotNotSupported{}
	// IONotSupported is a null IODriver interface
	IONotSupported = &ioNotSupported{}
	// StatsNotSupported is a null stats driver implementation. This can be used
	// by drivers that do not want to implement the stats interface.
	StatsNotSupported = &statsNotSupported{}
	// QuiesceNotSupported implements quiesce/unquiesce by returning not
	// supported error.
	QuiesceNotSupported = &quiesceNotSupported{}
	// CredsNotSupported implements credentials by returning not supported error
	CredsNotSupported = &credsNotSupported{}
	// CloudBackupNotSupported implements cloudBackupDriver by returning
	// Not supported error
	CloudBackupNotSupported = &cloudBackupNotSupported{}
	// CloudMigrateNotSupported implements cloudMigrateDriver by returning
	// Not supported error
	CloudMigrateNotSupported = &cloudMigrateNotSupported{}
	// FilesystemTrimNotSupported implements FilesystemTrimDriver by returning
	// Not supported error
	FilesystemTrimNotSupported = &filesystemTrimNotSupported{}
	// FilesystemCheckNotSupported implements FilesystemCheckDriver by returning
	// Not supported error
	FilesystemCheckNotSupported = &filesystemCheckNotSupported{}
)

type blockNotSupported struct{}

func (b *blockNotSupported) Attach(ctx context.Context, volumeID string, attachOptions map[string]string) (string, error) {
	return "", ErrNotSupported
}

func (b *blockNotSupported) Detach(ctx context.Context, volumeID string, options map[string]string) error {
	return ErrNotSupported
}

type snapshotNotSupported struct{}

func (s *snapshotNotSupported) Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator, noRetry bool) (string, error) {
	return "", ErrNotSupported
}

func (s *snapshotNotSupported) Restore(volumeID, snapshotID string) error {
	return ErrNotSupported
}

func (s *snapshotNotSupported) SnapshotGroup(groupID string, labels map[string]string, volumeIDs []string, deleteOnFailure bool) (*api.GroupSnapCreateResponse, error) {
	return nil, ErrNotSupported
}

type ioNotSupported struct{}

func (i *ioNotSupported) Read(volumeID string, buffer []byte, size uint64, offset int64) (int64, error) {
	return 0, ErrNotSupported
}

func (i *ioNotSupported) Write(volumeID string, buffer []byte, size uint64, offset int64) (int64, error) {
	return 0, ErrNotSupported
}

func (i *ioNotSupported) Flush(volumeID string) error {
	return ErrNotSupported
}

type statsNotSupported struct{}

// Stats returns stats
func (s *statsNotSupported) Stats(
	volumeID string,
	cumulative bool,
) (*api.Stats, error) {
	return nil, ErrNotSupported
}

// UsedSize returns allocated size
func (s *statsNotSupported) UsedSize(volumeID string) (uint64, error) {
	return 0, ErrNotSupported
}

// GetActiveRequests gets active requests
func (s *statsNotSupported) GetActiveRequests() (*api.ActiveRequests, error) {
	return nil, nil
}

// GetCapacityUsage gets exclusive and shared capacity
// usage of snap
func (s *statsNotSupported) CapacityUsage(
	ID string,
) (*api.CapacityUsageResponse, error) {
	return nil, ErrNotSupported
}

// VolumeUsageByNode returns capacity usage of all volumes/snaps belonging to
// a node
func (s *statsNotSupported) VolumeUsageByNode(
	nodeID string,
) (*api.VolumeUsageByNode, error) {
	return nil, ErrNotSupported
}

type quiesceNotSupported struct{}

func (s *quiesceNotSupported) Quiesce(
	volumeID string,
	timeoutSeconds uint64,
	quiesceID string,
) error {
	return ErrNotSupported
}

func (s *quiesceNotSupported) Unquiesce(volumeID string) error {
	return ErrNotSupported
}

type credsNotSupported struct{}

func (c *credsNotSupported) CredsCreate(
	params map[string]string,
) (string, error) {
	return "", ErrNotSupported
}

func (c *credsNotSupported) CredsUpdate(
	name string,
	params map[string]string,
) error {
	return ErrNotSupported
}

func (c *credsNotSupported) CredsDelete(
	uuid string,
) error {
	return ErrNotSupported
}

func (c *credsNotSupported) CredsEnumerate() (map[string]interface{}, error) {
	creds := make(map[string]interface{}, 0)
	return creds, ErrNotSupported
}

func (c *credsNotSupported) CredsValidate(
	uuid string,
) error {
	return ErrNotSupported
}

func (c *credsNotSupported) CredsDeleteReferences(
	uuid string,
) error {
	return ErrNotSupported
}

type cloudBackupNotSupported struct{}

func (cl *cloudBackupNotSupported) CloudBackupCreate(
	input *api.CloudBackupCreateRequest,
) (*api.CloudBackupCreateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupGroupCreate(
	input *api.CloudBackupGroupCreateRequest,
) (*api.CloudBackupGroupCreateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupRestore(
	input *api.CloudBackupRestoreRequest,
) (*api.CloudBackupRestoreResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupEnumerate(
	input *api.CloudBackupEnumerateRequest,
) (*api.CloudBackupEnumerateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupDelete(
	input *api.CloudBackupDeleteRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupDeleteAll(
	input *api.CloudBackupDeleteAllRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupStatus(
	input *api.CloudBackupStatusRequest,
) (*api.CloudBackupStatusResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupCatalog(
	input *api.CloudBackupCatalogRequest,
) (*api.CloudBackupCatalogResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupHistory(
	input *api.CloudBackupHistoryRequest,
) (*api.CloudBackupHistoryResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupStateChange(
	input *api.CloudBackupStateChangeRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupSchedCreate(
	input *api.CloudBackupSchedCreateRequest,
) (*api.CloudBackupSchedCreateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupSchedUpdate(
	input *api.CloudBackupSchedUpdateRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupGroupSchedUpdate(
	input *api.CloudBackupGroupSchedUpdateRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupGroupSchedCreate(
	input *api.CloudBackupGroupSchedCreateRequest,
) (*api.CloudBackupSchedCreateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupSchedDelete(
	input *api.CloudBackupSchedDeleteRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupSchedEnumerate() (*api.CloudBackupSchedEnumerateResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudBackupNotSupported) CloudBackupSize(
	input *api.SdkCloudBackupSizeRequest,
) (*api.SdkCloudBackupSizeResponse, error) {
	return nil, ErrNotSupported
}

type cloudMigrateNotSupported struct{}

func (cl *cloudMigrateNotSupported) CloudMigrateStart(request *api.CloudMigrateStartRequest) (*api.CloudMigrateStartResponse, error) {
	return nil, ErrNotSupported
}

func (cl *cloudMigrateNotSupported) CloudMigrateCancel(request *api.CloudMigrateCancelRequest) error {
	return ErrNotSupported
}
func (cl *cloudMigrateNotSupported) CloudMigrateStatus(request *api.CloudMigrateStatusRequest) (*api.CloudMigrateStatusResponse, error) {
	return nil, ErrNotSupported
}

type filesystemTrimNotSupported struct{}

func (cl *filesystemTrimNotSupported) FilesystemTrimStart(request *api.SdkFilesystemTrimStartRequest) (*api.SdkFilesystemTrimStartResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) FilesystemTrimStatus(request *api.SdkFilesystemTrimStatusRequest) (*api.SdkFilesystemTrimStatusResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) AutoFilesystemTrimStatus(request *api.SdkAutoFSTrimStatusRequest) (*api.SdkAutoFSTrimStatusResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) AutoFilesystemTrimUsage(request *api.SdkAutoFSTrimUsageRequest) (*api.SdkAutoFSTrimUsageResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) FilesystemTrimStop(request *api.SdkFilesystemTrimStopRequest) (*api.SdkFilesystemTrimStopResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) AutoFilesystemTrimPush(request *api.SdkAutoFSTrimPushRequest) (*api.SdkAutoFSTrimPushResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemTrimNotSupported) AutoFilesystemTrimPop(request *api.SdkAutoFSTrimPopRequest) (*api.SdkAutoFSTrimPopResponse, error) {
	return nil, ErrNotSupported
}

type filesystemCheckNotSupported struct{}

func (cl *filesystemCheckNotSupported) FilesystemCheckStart(request *api.SdkFilesystemCheckStartRequest) (*api.SdkFilesystemCheckStartResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemCheckNotSupported) FilesystemCheckStatus(request *api.SdkFilesystemCheckStatusRequest) (*api.SdkFilesystemCheckStatusResponse, error) {
	return nil, ErrNotSupported
}
func (cl *filesystemCheckNotSupported) FilesystemCheckStop(request *api.SdkFilesystemCheckStopRequest) (*api.SdkFilesystemCheckStopResponse, error) {
	return nil, ErrNotSupported
}
