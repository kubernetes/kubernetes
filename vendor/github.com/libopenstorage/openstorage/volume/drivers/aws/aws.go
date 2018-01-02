package aws

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"syscall"
	"time"

	"go.pedge.io/dlog"
	"github.com/libopenstorage/openstorage/pkg/proto/time"
	
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/opsworks"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/chaos"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers/common"
	"github.com/portworx/kvdb"
)

const (
	// Name of the driver
	Name = "aws"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_BLOCK
	// AwsDBKey for openstorage
	AwsDBKey = "OpenStorageAWSKey"
	// awsAccessKeyID identifier for authentication.
	awsAccessKeyID = "AWS_ACCESS_KEY_ID"
	// awsSecretAccessKey identifier for authentication.
	awsSecretAccessKey = "AWS_SECRET_ACCESS_KEY"
)

var (
	koStrayCreate = chaos.Add("aws", "create", "create in driver before DB")
	koStrayDelete = chaos.Add("aws", "delete", "create in driver before DB")
)

// Metadata for the driver
type Metadata struct {
	zone     string
	instance string
}

// Driver implements VolumeDriver interface
type Driver struct {
	volume.StatsDriver
	volume.StoreEnumerator
	volume.IODriver
	ops StorageOps
	md  *Metadata
}

// Init aws volume driver metadata.
func Init(params map[string]string) (volume.VolumeDriver, error) {
	zone, err := metadata("placement/availability-zone")
	if err != nil {
		return nil, err
	}
	instance, err := metadata("instance-id")
	if err != nil {
		return nil, err
	}
	dlog.Infof("AWS instance %v zone %v", instance, zone)

	accessKey, secretKey, err := authKeys(params)
	if err != nil {
		return nil, err
	}
	creds := credentials.NewStaticCredentials(accessKey, secretKey, "")
	region := zone[:len(zone)-1]
	ec2 := ec2.New(
		session.New(
			&aws.Config{
				Region:      &region,
				Credentials: creds,
			},
		),
	)
	d := &Driver{
		StatsDriver: volume.StatsNotSupported,
		ops:         NewEc2Storage(instance, ec2),
		md: &Metadata{
			zone:     zone,
			instance: instance,
		},
		IODriver:        volume.IONotSupported,
		StoreEnumerator: common.NewDefaultStoreEnumerator(Name, kvdb.Instance()),
	}
	return d, nil
}

// authKeys return authentication keys for this instance.
func authKeys(params map[string]string) (string, string, error) {
	accessKey, err := getAuthKey(awsAccessKeyID, params)
	if err != nil {
		return "", "", err
	}

	secretKey, err := getAuthKey(awsSecretAccessKey, params)
	if err != nil {
		return "", "", err
	}
	return accessKey, secretKey, nil
}

// getAuthKey retrieves specicified key from params or env var
func getAuthKey(key string, params map[string]string) (string, error) {
	val, ok := params[key]
	if !ok {
		if val = os.Getenv(key); len(val) == 0 {
			return "", fmt.Errorf("Authentication error: %v is not set", key)
		}
	}
	return val, nil
}

// mapCos translates a CoS specified in spec to a volume.
func mapCos(cos uint32) (*int64, *string) {
	var iops int64
	var volType string
	switch {
	case cos < 2:
		iops, volType = 0, opsworks.VolumeTypeGp2
	case cos < 7:
		iops, volType = 10000, opsworks.VolumeTypeIo1
	default:
		iops, volType = 20000, opsworks.VolumeTypeIo1
	}
	return &iops, &volType
}

// metadata retrieves instance metadata specified by key.
func metadata(key string) (string, error) {
	client := http.Client{Timeout: time.Second * 10}
	url := "http://169.254.169.254/latest/meta-data/" + key
	res, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		err = fmt.Errorf("Code %d returned for url %s", res.StatusCode, url)
		return "", fmt.Errorf("Error querying AWS metadata for key %s: %v", key, err)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", fmt.Errorf("Error querying AWS metadata for key %s: %v", key, err)
	}
	if len(body) == 0 {
		return "", fmt.Errorf("Failed to retrieve AWS metadata for key %s: %v", key, err)
	}
	return string(body), nil
}

// Name returns the name of the driver
func (d *Driver) Name() string {
	return Name
}

// Type returns the type of the driver
func (d *Driver) Type() api.DriverType {
	return Type
}

// Status returns the current status
func (d *Driver) Status() [][2]string {
	return [][2]string{}
}

// Create creates a new volume
func (d *Driver) Create(
	locator *api.VolumeLocator,
	source *api.Source,
	spec *api.VolumeSpec,
) (string, error) {
	var snapID *string
	// Spec size is in bytes, translate to GiB.
	sz := int64(spec.Size / (1024 * 1024 * 1024))
	iops, volType := mapCos(uint32(spec.Cos))
	if source != nil && string(source.Parent) != "" {
		id := string(source.Parent)
		snapID = &id
	}
	ec2Vol := &ec2.Volume{
		AvailabilityZone: &d.md.zone,
		VolumeType:       volType,
		SnapshotId:       snapID,
		Size:             &sz,
	}

	// Gp2 Volumes don't support the iops parameter
	if *volType != opsworks.VolumeTypeGp2 {
		ec2Vol.Iops = iops
	}
	vol, err := d.ops.Create(ec2Vol, locator.VolumeLabels)
	if err != nil {
		dlog.Warnf("Failed in CreateVolumeRequest :%v", err)
		return "", err
	}
	volume := common.NewVolume(
		*vol.VolumeId,
		api.FSType_FS_TYPE_EXT4,
		locator,
		source,
		spec,
	)
	err = d.UpdateVol(volume)
	if err != nil {
		return "", err
	}
	if _, err := d.Attach(volume.Id, nil); err != nil {
		return "", err
	}

	dlog.Infof("aws preparing volume %s...", *vol.VolumeId)
	if err := d.Format(volume.Id); err != nil {
		return "", err
	}
	if err := d.Detach(volume.Id, false); err != nil {
		return "", err
	}

	return volume.Id, err
}

// merge volume properties from aws into volume.
func (d *Driver) merge(v *api.Volume, aws *ec2.Volume) {
	v.AttachedOn = ""
	v.State = api.VolumeState_VOLUME_STATE_DETACHED
	v.DevicePath = ""

	switch *aws.State {
	case ec2.VolumeStateAvailable:
		v.Status = api.VolumeStatus_VOLUME_STATUS_UP
	case ec2.VolumeStateCreating, ec2.VolumeStateDeleting:
		v.State = api.VolumeState_VOLUME_STATE_PENDING
		v.Status = api.VolumeStatus_VOLUME_STATUS_DOWN
	case ec2.VolumeStateDeleted:
		v.State = api.VolumeState_VOLUME_STATE_DELETED
		v.Status = api.VolumeStatus_VOLUME_STATUS_DOWN
	case ec2.VolumeStateError:
		v.State = api.VolumeState_VOLUME_STATE_ERROR
		v.Status = api.VolumeStatus_VOLUME_STATUS_DOWN
	case ec2.VolumeStateInUse:
		v.Status = api.VolumeStatus_VOLUME_STATUS_UP
		if aws.Attachments != nil && len(aws.Attachments) != 0 {
			if aws.Attachments[0].InstanceId != nil {
				v.AttachedOn = *aws.Attachments[0].InstanceId
			}
			if aws.Attachments[0].State != nil {
				v.State = d.volumeState(aws.Attachments[0].State)
			}
			if aws.Attachments[0].Device != nil {
				v.DevicePath = *aws.Attachments[0].Device
			}
		}
	}
}

// Inspect insepcts a volume
func (d *Driver) Inspect(volumeIDs []string) ([]*api.Volume, error) {
	vols, err := d.StoreEnumerator.Inspect(volumeIDs)
	if err != nil {
		return nil, err
	}
	ids := make([]*string, len(vols))
	for i, v := range vols {
		id := v.Id
		ids[i] = &id
	}
	volumeMap, err := d.ops.Enumerate(ids, nil, "")
	if err != nil {
		return nil, err
	}
	if len(volumeMap) != 1 {
		return nil, fmt.Errorf("Inspect volumeMap mismatch")
	}
	for _, awsVols := range volumeMap {
		if len(awsVols) != len(vols) {
			return nil, fmt.Errorf("Inspect volume count mismatch")
		}
		for i, v := range awsVols {
			if string(vols[i].Id) != *v.VolumeId {
				d.merge(vols[i], v)
			}
		}
	}
	return vols, nil
}

func (d *Driver) Delete(volumeID string) error {
	if err := d.ops.Delete(volumeID); err != nil {
		return err
	}
	return d.DeleteVol(volumeID)
}

func (d *Driver) Snapshot(
	volumeID string,
	readonly bool,
	locator *api.VolumeLocator,
) (string, error) {
	vols, err := d.StoreEnumerator.Inspect([]string{volumeID})
	if err != nil {
		return "", err
	}
	if len(vols) != 1 {
		return "", fmt.Errorf("Failed to inspect %v len %v", volumeID, len(vols))
	}
	snap, err := d.ops.Snapshot(volumeID, readonly)
	if err != nil {
		return "", err
	}
	chaos.Now(koStrayCreate)
	vols[0].Id = *snap.SnapshotId
	vols[0].Source = &api.Source{Parent: volumeID}
	vols[0].Locator = locator
	vols[0].Ctime = prototime.Now()

	chaos.Now(koStrayCreate)
	if err = d.CreateVol(vols[0]); err != nil {
		return "", err
	}
	return vols[0].Id, nil
}

func (d *Driver) Restore(volumeID string, snapID string) error {
	// New volumes can be created from snapshot but existing volumes
	// cannot be restored to same volumeID.
	return volume.ErrNotSupported
}

func (d *Driver) Attach(
	volumeID string,
	attachOptions map[string]string,
) (string, error) {
	volume, err := d.GetVol(volumeID)
	if err != nil {
		return "", fmt.Errorf("Volume %s could not be located", volumeID)
	}
	path, err := d.ops.Attach(volumeID)
	if err != nil {
		return "", err
	}
	volume.DevicePath = path
	if err := d.UpdateVol(volume); err != nil {
		d.ops.Detach(volumeID)
		return "", err
	}
	return path, nil
}

func (d *Driver) volumeState(ec2VolState *string) api.VolumeState {
	if ec2VolState == nil {
		return api.VolumeState_VOLUME_STATE_DETACHED
	}
	switch *ec2VolState {
	case ec2.VolumeAttachmentStateAttached:
		return api.VolumeState_VOLUME_STATE_ATTACHED
	case ec2.VolumeAttachmentStateDetached:
		return api.VolumeState_VOLUME_STATE_DETACHED
	case ec2.VolumeAttachmentStateAttaching, ec2.VolumeAttachmentStateDetaching:
		return api.VolumeState_VOLUME_STATE_PENDING
	default:
		dlog.Warnf("Failed to translate EC2 volume status %v", ec2VolState)
	}
	return api.VolumeState_VOLUME_STATE_ERROR
}

func (d *Driver) Format(volumeID string) error {
	volume, err := d.GetVol(volumeID)
	if err != nil {
		return fmt.Errorf("Failed to locate volume %q", volumeID)
	}

	// XXX: determine mount state
	awsVol, err := d.ops.Inspect([]*string{&volumeID})
	if err != nil {
		return err
	}
	if len(awsVol) != 1 {
		return fmt.Errorf("Failed to inspect volume %v", volumeID)
	}
	devicePath, err := d.ops.DevicePath(awsVol[0])
	if err != nil {
		return err
	}
	cmd := "/sbin/mkfs." + volume.Spec.Format.SimpleString()
	o, err := exec.Command(cmd, devicePath).Output()
	if err != nil {
		dlog.Warnf("Failed to run command %v %v: %v", cmd, devicePath, o)
		return err
	}
	volume.Format = volume.Spec.Format
	return d.UpdateVol(volume)
}

func (d *Driver) Detach(volumeID string, unmountBeforeDetach bool) error {
	if err := d.ops.Detach(volumeID); err != nil {
		return err
	}
	volume, err := d.GetVol(volumeID)
	if err != nil {
		dlog.Warnf("Volume %s could not be located, attempting to detach anyway", volumeID)
	} else {
		volume.DevicePath = ""
		if err := d.UpdateVol(volume); err != nil {
			dlog.Warnf("Failed to update volume", volumeID)
		}
	}
	return nil
}

func (d *Driver) MountedAt(mountpath string) string {
	return ""
}

func (d *Driver) Mount(volumeID string, mountpath string) error {
	volume, err := d.GetVol(volumeID)
	if err != nil {
		return fmt.Errorf("Failed to locate volume %q", volumeID)
	}
	awsVol, err := d.ops.Inspect([]*string{&volumeID})
	if err != nil {
		return err
	}
	if len(awsVol) != 1 {
		return fmt.Errorf("Failed to inspect volume %v", volumeID)
	}
	devicePath, err := d.ops.DevicePath(awsVol[0])
	if err != nil {
		return err
	}
	err = syscall.Mount(devicePath, mountpath, volume.Spec.Format.SimpleString(), 0, "")
	if err != nil {
		return err
	}
	return nil
}

func (d *Driver) Unmount(volumeID string, mountpath string) error {
	// XXX:  determine if valid mount path
	err := syscall.Unmount(mountpath, 0)
	return err
}

func (d *Driver) Shutdown() {
	dlog.Printf("%s Shutting down", Name)
}

func (d *Driver) Set(volumeID string, locator *api.VolumeLocator, spec *api.VolumeSpec) error {
	return volume.ErrNotSupported
}
