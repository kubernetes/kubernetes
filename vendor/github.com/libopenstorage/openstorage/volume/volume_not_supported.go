package volume

import (
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
	//  Not supported error
	CloudBackupNotSupported = &cloudBackupNotSupported{}
)

type blockNotSupported struct{}

func (b *blockNotSupported) Attach(volumeID string, attachOptions map[string]string) (string, error) {
	return "", ErrNotSupported
}

func (b *blockNotSupported) Detach(volumeID string, options map[string]string) error {
	return ErrNotSupported
}

type snapshotNotSupported struct{}

func (s *snapshotNotSupported) Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator) (string, error) {
	return "", ErrNotSupported
}

func (s *snapshotNotSupported) Restore(volumeID, snapshotID string) error {
	return ErrNotSupported
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

type cloudBackupNotSupported struct{}

func (cl *cloudBackupNotSupported) Backup(
	input *api.BackupRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) BackupRestore(
	input *api.BackupRestoreRequest,
) *api.BackupRestoreResponse {
	response := api.BackupRestoreResponse{
		RestoreErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupEnumerate(
	input *api.BackupEnumerateRequest,
) *api.BackupEnumerateResponse {
	response := api.BackupEnumerateResponse{
		EnumerateErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupDelete(
	input *api.BackupDeleteRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) BackupStatus(
	input *api.BackupStsRequest,
) *api.BackupStsResponse {
	response := api.BackupStsResponse{
		StsErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupCatalogue(
	input *api.BackupCatalogueRequest,
) *api.BackupCatalogueResponse {
	response := api.BackupCatalogueResponse{
		CatalogueErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupHistory(
	input *api.BackupHistoryRequest,
) *api.BackupHistoryResponse {
	response := api.BackupHistoryResponse{
		HistoryErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupStateChange(
	input *api.BackupStateChangeRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) BackupSchedCreate(
	input *api.BackupScheduleInfo,
) *api.BackupSchedResponse {
	response := api.BackupSchedResponse{
		SchedCreateErr: ErrNotSupported.Error(),
	}
	return &response
}

func (cl *cloudBackupNotSupported) BackupSchedDelete(
	input *api.BackupSchedDeleteRequest,
) error {
	return ErrNotSupported
}

func (cl *cloudBackupNotSupported) BackupSchedEnumerate() *api.BackupSchedEnumerateResponse {
	response := api.BackupSchedEnumerateResponse{
		SchedEnumerateErr: ErrNotSupported.Error(),
	}
	return &response
}
