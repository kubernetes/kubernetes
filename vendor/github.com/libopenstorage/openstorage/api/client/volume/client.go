package volume

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/client"
	"github.com/libopenstorage/openstorage/volume"
	"io"
	"io/ioutil"
	"strconv"
)

const (
	graphPath  = "/graph"
	volumePath = "/osd-volumes"
	snapPath   = "/osd-snapshot"
	credsPath  = "/osd-creds"
	backupPath = "/osd-backup"
)

type volumeClient struct {
	volume.IODriver
	c *client.Client
}

func newVolumeClient(c *client.Client) volume.VolumeDriver {
	return &volumeClient{volume.IONotSupported, c}
}

// String description of this driver.
func (v *volumeClient) Name() string {
	return "VolumeDriver"
}

func (v *volumeClient) Type() api.DriverType {
	// Block drivers implement the superset.
	return api.DriverType_DRIVER_TYPE_BLOCK
}

func (v *volumeClient) GraphDriverCreate(id string, parent string) error {
	response := ""
	if err := v.c.Put().Resource(graphPath + "/create").Instance(id).Do().Unmarshal(&response); err != nil {
		return err
	}
	if response != id {
		return fmt.Errorf("Invalid response: %s", response)
	}
	return nil
}

func (v *volumeClient) GraphDriverRemove(id string) error {
	response := ""
	if err := v.c.Put().Resource(graphPath + "/remove").Instance(id).Do().Unmarshal(&response); err != nil {
		return err
	}
	if response != id {
		return fmt.Errorf("Invalid response: %s", response)
	}
	return nil
}

func (v *volumeClient) GraphDriverGet(id string, mountLabel string) (string, error) {
	response := ""
	if err := v.c.Get().Resource(graphPath + "/inspect").Instance(id).Do().Unmarshal(&response); err != nil {
		return "", err
	}
	return response, nil
}

func (v *volumeClient) GraphDriverRelease(id string) error {
	response := ""
	if err := v.c.Put().Resource(graphPath + "/release").Instance(id).Do().Unmarshal(&response); err != nil {
		return err
	}
	if response != id {
		return fmt.Errorf("Invalid response: %v", response)
	}
	return nil
}

func (v *volumeClient) GraphDriverExists(id string) bool {
	response := false
	v.c.Get().Resource(graphPath + "/exists").Instance(id).Do().Unmarshal(&response)
	return response
}

func (v *volumeClient) GraphDriverDiff(id string, parent string) io.Writer {
	body, _ := v.c.Get().Resource(graphPath + "/diff?id=" + id + "&parent=" + parent).Do().Body()
	return bytes.NewBuffer(body)
}

func (v *volumeClient) GraphDriverChanges(id string, parent string) ([]api.GraphDriverChanges, error) {
	var changes []api.GraphDriverChanges
	err := v.c.Get().Resource(graphPath + "/changes").Instance(id).Do().Unmarshal(&changes)
	return changes, err
}

func (v *volumeClient) GraphDriverApplyDiff(id string, parent string, diff io.Reader) (int, error) {
	b, err := ioutil.ReadAll(diff)
	if err != nil {
		return 0, err
	}
	response := 0
	if err = v.c.Put().Resource(graphPath + "/diff?id=" + id + "&parent=" + parent).Instance(id).Body(b).Do().Unmarshal(&response); err != nil {
		return 0, err
	}
	return response, nil
}

func (v *volumeClient) GraphDriverDiffSize(id string, parent string) (int, error) {
	size := 0
	err := v.c.Get().Resource(graphPath + "/diffsize").Instance(id).Do().Unmarshal(&size)
	return size, err
}

// Create a new Vol for the specific volume spev.c.
// It returns a system generated VolumeID that uniquely identifies the volume
func (v *volumeClient) Create(locator *api.VolumeLocator, source *api.Source,
	spec *api.VolumeSpec) (string, error) {
	response := &api.VolumeCreateResponse{}
	request := &api.VolumeCreateRequest{
		Locator: locator,
		Source:  source,
		Spec:    spec,
	}
	if err := v.c.Post().Resource(volumePath).Body(request).Do().Unmarshal(response); err != nil {
		return "", err
	}
	if response.VolumeResponse != nil && response.VolumeResponse.Error != "" {
		return "", errors.New(response.VolumeResponse.Error)
	}
	return response.Id, nil
}

// Status diagnostic information
func (v *volumeClient) Status() [][2]string {
	return [][2]string{}
}

// Inspect specified volumes.
// Errors ErrEnoEnt may be returned.
func (v *volumeClient) Inspect(ids []string) ([]*api.Volume, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var volumes []*api.Volume
	request := v.c.Get().Resource(volumePath)
	for _, id := range ids {
		request.QueryOption(api.OptVolumeID, id)
	}
	if err := request.Do().Unmarshal(&volumes); err != nil {
		return nil, err
	}
	return volumes, nil
}

// Delete volume.
// Errors ErrEnoEnt, ErrVolHasSnaps may be returned.
func (v *volumeClient) Delete(volumeID string) error {
	response := &api.VolumeResponse{}
	if err := v.c.Delete().Resource(volumePath).Instance(volumeID).Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// Snap specified volume. IO to the underlying volume should be quiesced before
// calling this function.
// Errors ErrEnoEnt may be returned
func (v *volumeClient) Snapshot(volumeID string, readonly bool,
	locator *api.VolumeLocator) (string, error) {
	response := &api.SnapCreateResponse{}
	request := &api.SnapCreateRequest{
		Id:       volumeID,
		Readonly: readonly,
		Locator:  locator,
	}
	if err := v.c.Post().Resource(snapPath).Body(request).Do().Unmarshal(response); err != nil {
		return "", err
	}
	// TODO(pedge): this probably should not be embedded in this way
	if response.VolumeCreateResponse != nil &&
		response.VolumeCreateResponse.VolumeResponse != nil &&
		response.VolumeCreateResponse.VolumeResponse.Error != "" {
		return "", errors.New(
			response.VolumeCreateResponse.VolumeResponse.Error)
	}
	if response.VolumeCreateResponse != nil {
		return response.VolumeCreateResponse.Id, nil
	}
	return "", nil
}

// Restore specified volume to given snapshot state
func (v *volumeClient) Restore(volumeID string, snapID string) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(snapPath + "/restore").Instance(volumeID)
	req.QueryOption(api.OptSnapID, snapID)

	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// Stats for specified volume.
// Errors ErrEnoEnt may be returned
func (v *volumeClient) Stats(
	volumeID string,
	cumulative bool,
) (*api.Stats, error) {
	stats := &api.Stats{}
	req := v.c.Get().Resource(volumePath + "/stats").Instance(volumeID)
	req.QueryOption(api.OptCumulative, strconv.FormatBool(cumulative))

	err := req.Do().Unmarshal(stats)
	return stats, err

}

// UsedSize returns allocated volume size.
// Errors ErrEnoEnt may be returned
func (v *volumeClient) UsedSize(
	volumeID string,
) (uint64, error) {
	var usedSize uint64
	req := v.c.Get().Resource(volumePath + "/usedsize").Instance(volumeID)
	err := req.Do().Unmarshal(&usedSize)
	return usedSize, err
}

// Active Requests on all volume.
func (v *volumeClient) GetActiveRequests() (*api.ActiveRequests, error) {

	requests := &api.ActiveRequests{}
	resp := v.c.Get().Resource(volumePath + "/requests").Instance("vol_id").Do()

	if resp.Error() != nil {
		return nil, resp.FormatError()
	}

	if err := resp.Unmarshal(requests); err != nil {
		return nil, err
	}

	return requests, nil
}

// Shutdown and cleanup.
func (v *volumeClient) Shutdown() {}

// Enumerate volumes that map to the volumeLocator. Locator fields may be regexp.
// If locator fields are left blank, this will return all volumes.
func (v *volumeClient) Enumerate(locator *api.VolumeLocator,
	labels map[string]string) ([]*api.Volume, error) {
	var volumes []*api.Volume
	req := v.c.Get().Resource(volumePath)
	if locator.Name != "" {
		req.QueryOption(api.OptName, locator.Name)
	}
	if len(locator.VolumeLabels) != 0 {
		req.QueryOptionLabel(api.OptLabel, locator.VolumeLabels)
	}
	if len(labels) != 0 {
		req.QueryOptionLabel(api.OptConfigLabel, labels)
	}
	resp := req.Do()
	if resp.Error() != nil {
		return nil, resp.FormatError()
	}
	if err := resp.Unmarshal(&volumes); err != nil {
		return nil, err
	}

	return volumes, nil
}

// Enumerate snaps for specified volume
// Count indicates the number of snaps populated.
func (v *volumeClient) SnapEnumerate(ids []string,
	snapLabels map[string]string) ([]*api.Volume, error) {
	var volumes []*api.Volume
	request := v.c.Get().Resource(snapPath)
	for _, id := range ids {
		request.QueryOption(api.OptVolumeID, id)
	}
	if len(snapLabels) != 0 {
		request.QueryOptionLabel(api.OptLabel, snapLabels)
	}
	if err := request.Do().Unmarshal(&volumes); err != nil {
		return nil, err
	}
	return volumes, nil
}

// Attach map device to the host.
// On success the devicePath specifies location where the device is exported
// Errors ErrEnoEnt, ErrVolAttached may be returned.
func (v *volumeClient) Attach(volumeID string, attachOptions map[string]string) (string, error) {
	response, err := v.doVolumeSetGetResponse(
		volumeID,
		&api.VolumeSetRequest{
			Action: &api.VolumeStateAction{
				Attach: api.VolumeActionParam_VOLUME_ACTION_PARAM_ON,
			},
			Options: attachOptions,
		},
	)
	if err != nil {
		return "", err
	}
	if response.Volume != nil {
		if response.Volume.Spec.Encrypted {
			return response.Volume.SecureDevicePath, nil
		} else {
			return response.Volume.DevicePath, nil
		}
	}
	return "", nil
}

// Detach device from the host.
// Errors ErrEnoEnt, ErrVolDetached may be returned.
func (v *volumeClient) Detach(volumeID string, options map[string]string) error {
	return v.doVolumeSet(
		volumeID,
		&api.VolumeSetRequest{
			Action: &api.VolumeStateAction{
				Attach: api.VolumeActionParam_VOLUME_ACTION_PARAM_OFF,
			},
			Options: options,
		},
	)
}

func (v *volumeClient) MountedAt(mountPath string) string {
	return ""
}

// Mount volume at specified path
// Errors ErrEnoEnt, ErrVolDetached may be returned.
func (v *volumeClient) Mount(volumeID string, mountPath string, options map[string]string) error {
	return v.doVolumeSet(
		volumeID,
		&api.VolumeSetRequest{
			Action: &api.VolumeStateAction{
				Mount:     api.VolumeActionParam_VOLUME_ACTION_PARAM_ON,
				MountPath: mountPath,
			},
			Options: options,
		},
	)
}

// Unmount volume at specified path
// Errors ErrEnoEnt, ErrVolDetached may be returned.
func (v *volumeClient) Unmount(volumeID string, mountPath string, options map[string]string) error {
	return v.doVolumeSet(
		volumeID,
		&api.VolumeSetRequest{
			Action: &api.VolumeStateAction{
				Mount:     api.VolumeActionParam_VOLUME_ACTION_PARAM_OFF,
				MountPath: mountPath,
			},
			Options: options,
		},
	)
}

// Update volume
func (v *volumeClient) Set(volumeID string, locator *api.VolumeLocator,
	spec *api.VolumeSpec) error {
	return v.doVolumeSet(
		volumeID,
		&api.VolumeSetRequest{
			Locator: locator,
			Spec:    spec,
		},
	)
}

func (v *volumeClient) doVolumeSet(volumeID string,
	request *api.VolumeSetRequest) error {
	_, err := v.doVolumeSetGetResponse(volumeID, request)
	return err
}

func (v *volumeClient) doVolumeSetGetResponse(volumeID string,
	request *api.VolumeSetRequest) (*api.VolumeSetResponse, error) {
	response := &api.VolumeSetResponse{}
	if err := v.c.Put().Resource(volumePath).Instance(volumeID).Body(request).Do().Unmarshal(response); err != nil {
		return nil, err
	}
	if response.VolumeResponse != nil && response.VolumeResponse.Error != "" {
		return nil, errors.New(response.VolumeResponse.Error)
	}
	return response, nil
}

// Quiesce quiesces volume i/o
func (v *volumeClient) Quiesce(
	volumeID string,
	timeoutSec uint64,
	quiesceID string,
) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(volumePath + "/quiesce").Instance(volumeID)
	req.QueryOption(api.OptTimeoutSec, strconv.FormatUint(timeoutSec, 10))
	req.QueryOption(api.OptQuiesceID, quiesceID)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// Unquiesce un-quiesces volume i/o
func (v *volumeClient) Unquiesce(volumeID string) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(volumePath + "/unquiesce").Instance(volumeID)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// CredsEnumerate enumerates configured credentials in the cluster
func (v *volumeClient) CredsEnumerate() (map[string]interface{}, error) {
	creds := make(map[string]interface{}, 0)
	err := v.c.Get().Resource(credsPath).Do().Unmarshal(&creds)
	return creds, err
}

// CredsCreate creates credentials for a given cloud provider
func (v *volumeClient) CredsCreate(params map[string]string) (string, error) {
	response := api.CredCreateResponse{}
	request := &api.CredCreateRequest{
		InputParams: params,
	}
	err := v.c.Post().Resource(credsPath).Body(request).Do().Unmarshal(&response)
	if err == nil {
		if response.CredErr != "" {
			err = errors.New(response.CredErr)
		}
	}
	return response.UUID, err
}

// CredsDelete deletes the credential with given UUID
func (v *volumeClient) CredsDelete(uuid string) error {
	response := &api.VolumeResponse{}
	req := v.c.Delete().Resource(credsPath).Instance(uuid)
	err := req.Do().Unmarshal(&response)
	if err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// CredsValidate validates the credential by accessuing the cloud
// provider with the given credential
func (v *volumeClient) CredsValidate(uuid string) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(credsPath + "/validate").Instance(uuid)
	err := req.Do().Unmarshal(&response)
	if err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// Backup uploads snapshot of a volume to cloud
func (v *volumeClient) Backup(
	input *api.BackupRequest,
) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(backupPath).Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// BackupRestore downloads a cloud backup to a newly created volume
func (v *volumeClient) BackupRestore(
	input *api.BackupRestoreRequest,
) *api.BackupRestoreResponse {
	response := &api.BackupRestoreResponse{}
	req := v.c.Post().Resource(backupPath + "/restore").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.RestoreErr = err.Error()
	}
	return response
}

// BackupEnumerate lists the backups for a given cluster/credential/volumeID
func (v *volumeClient) BackupEnumerate(
	input *api.BackupEnumerateRequest,
) *api.BackupEnumerateResponse {
	response := &api.BackupEnumerateResponse{}
	req := v.c.Get().Resource(backupPath).Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.EnumerateErr = err.Error()
	}
	return response

}

// BackupDelete deletes the backups in cloud
func (v *volumeClient) BackupDelete(
	input *api.BackupDeleteRequest,
) error {
	response := &api.VolumeResponse{}
	req := v.c.Delete().Resource(backupPath).Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// BackupStatus indicates the most recent status of backup/restores
func (v *volumeClient) BackupStatus(
	input *api.BackupStsRequest,
) *api.BackupStsResponse {
	response := &api.BackupStsResponse{}
	req := v.c.Post().Resource(backupPath + "/status").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.StsErr = err.Error()
	}
	return response
}

// BackupCatalogue displays listing of backup content
func (v *volumeClient) BackupCatalogue(
	input *api.BackupCatalogueRequest,
) *api.BackupCatalogueResponse {
	response := &api.BackupCatalogueResponse{}
	req := v.c.Get().Resource(backupPath + "/catalogue").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.CatalogueErr = err.Error()
	}
	return response
}

// History displays past backup/restore operations in the cluster
func (v *volumeClient) BackupHistory(
	input *api.BackupHistoryRequest,
) *api.BackupHistoryResponse {
	response := &api.BackupHistoryResponse{}
	req := v.c.Get().Resource(backupPath + "/history").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.HistoryErr = err.Error()
	}
	return response

}

// ChangeBackupState allows a current backup
// state transisions(pause/resume/stop)
func (v *volumeClient) BackupStateChange(
	input *api.BackupStateChangeRequest,
) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(backupPath + "/statechange").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// BackupSchedCreate for a volume creates a schedule to backup volume to cloud
func (v *volumeClient) BackupSchedCreate(
	input *api.BackupScheduleInfo,
) *api.BackupSchedResponse {
	response := &api.BackupSchedResponse{}
	req := v.c.Post().Resource(backupPath + "/schedcreate").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		response.SchedCreateErr = err.Error()
	}
	return response

}

// BackupSchedDelete delete a volume's cloud backup-schedule
func (v *volumeClient) BackupSchedDelete(
	input *api.BackupSchedDeleteRequest,
) error {
	response := &api.VolumeResponse{}
	req := v.c.Post().Resource(backupPath + "/scheddelete").Body(input)
	if err := req.Do().Unmarshal(response); err != nil {
		return err
	}
	if response.Error != "" {
		return errors.New(response.Error)
	}
	return nil
}

// BackupSchedEnumerate enumerates the configured backup-schedules in the cluster
func (v *volumeClient) BackupSchedEnumerate() *api.BackupSchedEnumerateResponse {
	response := &api.BackupSchedEnumerateResponse{}
	req := v.c.Get().Resource(backupPath + "/schedenumerate")
	if err := req.Do().Unmarshal(response); err != nil {
		response.SchedEnumerateErr = err.Error()
	}
	return response
}
