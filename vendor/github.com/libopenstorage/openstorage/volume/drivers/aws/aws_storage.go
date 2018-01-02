package aws

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/opsworks"
	"github.com/libopenstorage/openstorage/api"
)

type ec2Ops struct {
	instance string
	ec2      *ec2.EC2
	mutex    sync.Mutex
}

const (
	SetIdentifierNone = "None"
)

// Custom AWS volume error codes.
const (
	_ = iota + 5000
	ErrVolDetached
	ErrVolInval
	ErrVolAttachedOnRemoteNode
)

// StorageError error returned for AWS volumes
type StorageError struct {
	// Code is one of AWS volume error codes.
	Code int
	// Msg is human understandable error message.
	Msg string
	// Instance provides more information on the error.
	Instance string
}

// StorageOps interface to perform basic operations on aws.
type StorageOps interface {
	// Create volume based on input template volume.
	// Apply labels as tags on EBS volumes
	Create(template *ec2.Volume, labels map[string]string) (*ec2.Volume, error)
	// Attach volumeID.
	// Return attach path.
	Attach(volumeID string) (string, error)
	// Detach volumeID.
	Detach(volumeID string) error
	// Delete volumeID.
	Delete(volumeID string) error
	// Inspect volumes specified by volumeID
	Inspect(volumeIds []*string) ([]*ec2.Volume, error)
	// DeviceMappings returns map[local_volume_path]->aws volume ID
	DeviceMappings() (map[string]string, error)
	// Enumerate EBS volumes that match given filters.  Organize them into
	// sets identified by setIdentifier.
	// labels can be nil, setIdentifier can be empty string.
	Enumerate(volumeIds []*string,
		labels map[string]string,
		setIdentifier string,
	) (map[string][]*ec2.Volume, error)
	// DevicePath for attached EBS volume.
	DevicePath(volume *ec2.Volume) (string, error)
	// Snapshot EBS volume
	Snapshot(volumeID string, readonly bool) (*ec2.Snapshot, error)
	// ApplyTags
	ApplyTags(v *ec2.Volume, labels map[string]string) error
	// Tags
	Tags(v *ec2.Volume) map[string]string
}

func NewStorageError(code int, msg string, instance string) error {
	return &StorageError{Code: code, Msg: msg, Instance: instance}
}

func (e *StorageError) Error() string {
	return e.Msg
}

func NewEc2Storage(instance string, ec2 *ec2.EC2) StorageOps {
	return &ec2Ops{
		instance: instance,
		ec2:      ec2,
	}
}

func (s *ec2Ops) mapVolumeType(awsVol *ec2.Volume) api.StorageMedium {
	switch *awsVol.VolumeType {
	case opsworks.VolumeTypeGp2:
		return api.StorageMedium_STORAGE_MEDIUM_SSD
	case opsworks.VolumeTypeIo1:
		return api.StorageMedium_STORAGE_MEDIUM_NVME
	case opsworks.VolumeTypeStandard:
		return api.StorageMedium_STORAGE_MEDIUM_MAGNETIC
	}
	return api.StorageMedium_STORAGE_MEDIUM_MAGNETIC
}

func (s *ec2Ops) filters(
	labels map[string]string,
	keys []string,
) []*ec2.Filter {
	if len(labels) == 0 {
		return nil
	}
	f := make([]*ec2.Filter, len(labels)+len(keys))
	i := 0
	for k, v := range labels {
		s := string("tag:") + k
		value := v
		f[i] = &ec2.Filter{Name: &s, Values: []*string{&value}}
		i++
	}
	for _, k := range keys {
		s := string("tag-key:") + k
		f[i] = &ec2.Filter{Name: &s}
		i++
	}
	return f
}

func (s *ec2Ops) tags(labels map[string]string) []*ec2.Tag {
	if len(labels) == 0 {
		return nil
	}
	t := make([]*ec2.Tag, len(labels))
	i := 0
	for k, v := range labels {
		key := k
		value := v
		t[i] = &ec2.Tag{Key: &key, Value: &value}
		i++
	}
	return t
}

func (s *ec2Ops) waitStatus(id string, desired string) error {
	request := &ec2.DescribeVolumesInput{VolumeIds: []*string{&id}}
	actual := ""

	for retries, maxRetries := 0, 10; actual != desired && retries < maxRetries; retries++ {
		awsVols, err := s.ec2.DescribeVolumes(request)
		if err != nil {
			return err
		}
		if len(awsVols.Volumes) != 1 {
			return fmt.Errorf("expected one volume %v got %v",
				id, len(awsVols.Volumes))
		}
		if awsVols.Volumes[0].State == nil {
			return fmt.Errorf("Nil volume state for %v", id)
		}
		actual = *awsVols.Volumes[0].State
		if actual == desired {
			break
		}
		time.Sleep(3 * time.Second)
	}
	if actual != desired {
		return fmt.Errorf(
			"Volume %v did not transition to %v current state %v",
			id, desired, actual)
	}
	return nil
}

func (s *ec2Ops) waitAttachmentStatus(
	volumeID string,
	desired string,
	timeout time.Duration,
) (*ec2.Volume, error) {
	id := volumeID
	request := &ec2.DescribeVolumesInput{VolumeIds: []*string{&id}}
	actual := ""
	interval := 2 * time.Second
	fmt.Printf("Waiting for state transition to %q", desired)

	var outVol *ec2.Volume
	for elapsed, runs := 0*time.Second, 0; actual != desired && elapsed < timeout; elapsed += interval {
		awsVols, err := s.ec2.DescribeVolumes(request)
		if err != nil {
			return nil, err
		}
		if len(awsVols.Volumes) != 1 {
			return nil, fmt.Errorf("expected one volume %v got %v",
				volumeID, len(awsVols.Volumes))
		}
		outVol = awsVols.Volumes[0]
		awsAttachment := awsVols.Volumes[0].Attachments
		if awsAttachment == nil || len(awsAttachment) == 0 {
			actual = ec2.VolumeAttachmentStateDetached
			if actual == desired {
				break
			}
			return nil, fmt.Errorf("Nil attachment state for %v", volumeID)
		}
		actual = *awsAttachment[0].State
		if actual == desired {
			break
		}
		time.Sleep(interval)
		if (runs % 10) == 0 {
			fmt.Print(".")
		}
	}
	fmt.Printf("\n")
	if actual != desired {
		return nil, fmt.Errorf("Volume %v failed to transition to  %v current state %v",
			volumeID, desired, actual)
	}
	return outVol, nil
}

func (s *ec2Ops) ApplyTags(
	v *ec2.Volume,
	labels map[string]string,
) error {
	req := &ec2.CreateTagsInput{
		Resources: []*string{v.VolumeId},
		Tags:      s.tags(labels),
	}
	_, err := s.ec2.CreateTags(req)
	return err
}

func (s *ec2Ops) matchTag(tag *ec2.Tag, match string) bool {
	return tag.Key != nil &&
		tag.Value != nil &&
		len(*tag.Key) != 0 &&
		len(*tag.Value) != 0 &&
		*tag.Key == match
}

func (s *ec2Ops) addResource(
	sets map[string][]*ec2.Volume,
	vol *ec2.Volume,
	key string,
) {
	if s, ok := sets[key]; ok {
		sets[key] = append(s, vol)
	} else {
		sets[key] = []*ec2.Volume{vol}
	}
}

func (s *ec2Ops) DeviceMappings() (map[string]string, error) {
	instance, err := s.describe()
	if err != nil {
		return nil, err
	}
	devPrefix := "/dev/sd"
	m := make(map[string]string)
	for _, d := range instance.BlockDeviceMappings {
		if d.DeviceName != nil && d.Ebs != nil && d.Ebs.VolumeId != nil {
			devName := *d.DeviceName
			// Per AWS docs EC instances have the root mounted at
			// /dev/sda1, this label should be skipped
			if devName == "/dev/sda1" {
				continue
			}
			// AWS EBS volumes get mapped from /dev/sdN -->/dev/xvdN
			if strings.HasPrefix(devName, devPrefix) {
				devName = "/dev/xvd" + devName[len(devPrefix):]
			}
			m[devName] = *d.Ebs.VolumeId
		}
	}
	return m, nil
}

// describe current instance.
func (s *ec2Ops) describe() (*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{&s.instance},
	}
	out, err := s.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(out.Reservations) != 1 {
		return nil, fmt.Errorf("DescribeInstances(%v) returned %v reservations, expect 1",
			s.instance, len(out.Reservations))
	}
	if len(out.Reservations[0].Instances) != 1 {
		return nil, fmt.Errorf("DescribeInstances(%v) returned %v Reservations, expect 1",
			s.instance, len(out.Reservations[0].Instances))
	}
	return out.Reservations[0].Instances[0], nil
}

// freeDevices returns list of available device IDs.
func (s *ec2Ops) freeDevices() ([]string, error) {
	initial := []byte("fghijklmnop")
	self, err := s.describe()
	if err != nil {
		return nil, err
	}
	devPrefix := "/dev/sd"
	for _, dev := range self.BlockDeviceMappings {
		if dev.DeviceName == nil {
			return nil, fmt.Errorf("Nil device name")
		}
		devName := *dev.DeviceName

		// per AWS docs EC instances have the root mounted at /dev/sda1,
		// this label should be skipped
		if devName == "/dev/sda1" {
			continue
		}
		if !strings.HasPrefix(devName, devPrefix) {
			devPrefix = "/dev/xvd"
			if !strings.HasPrefix(devName, devPrefix) {
				return nil, fmt.Errorf("bad device name %q", devName)
			}
		}
		letter := devName[len(devPrefix):]
		if len(letter) != 1 {
			return nil, fmt.Errorf("too many letters %q", devName)
		}

		// Reset devPrefix for next devices
		devPrefix = "/dev/sd"

		index := letter[0] - 'f'
		if index > ('p' - 'f') {
			continue
		}
		initial[index] = '0'
	}
	free := make([]string, len(initial))
	count := 0
	for _, b := range initial {
		if b != '0' {
			free[count] = devPrefix + string(b)
			count++
		}
	}
	if count == 0 {
		return nil, fmt.Errorf("No more free devices")
	}
	return free[:count], nil
}

func (s *ec2Ops) rollbackCreate(id string, createErr error) error {
	logrus.Warnf("Rollback create volume %v, Error %v", id, createErr)
	err := s.Delete(id)
	if err != nil {
		logrus.Warnf("Rollback failed volume %v, Error %v", id, err)
	}
	return createErr
}

func (s *ec2Ops) deleted(v *ec2.Volume) bool {
	return *v.State == ec2.VolumeStateDeleting ||
		*v.State == ec2.VolumeStateDeleted
}

func (s *ec2Ops) available(v *ec2.Volume) bool {
	return *v.State == ec2.VolumeStateAvailable
}

func (s *ec2Ops) Inspect(volumeIds []*string) ([]*ec2.Volume, error) {
	req := &ec2.DescribeVolumesInput{VolumeIds: volumeIds}
	awsVols, err := s.ec2.DescribeVolumes(req)
	if err != nil {
		return nil, err
	}
	return awsVols.Volumes, nil
}

func (s *ec2Ops) Tags(v *ec2.Volume) map[string]string {
	labels := make(map[string]string)
	for _, tag := range v.Tags {
		labels[*tag.Key] = *tag.Value
	}
	return labels
}

func (s *ec2Ops) Enumerate(
	volumeIds []*string,
	labels map[string]string,
	setIdentifier string,
) (map[string][]*ec2.Volume, error) {

	sets := make(map[string][]*ec2.Volume)

	// Enumerate all volumes that have same labels.
	f := s.filters(labels, nil)
	req := &ec2.DescribeVolumesInput{Filters: f, VolumeIds: volumeIds}
	awsVols, err := s.ec2.DescribeVolumes(req)
	if err != nil {
		return nil, err
	}

	// Volume sets are identified by volumes with the same setIdentifer.
	found := false
	for _, vol := range awsVols.Volumes {
		if s.deleted(vol) {
			continue
		}
		if len(setIdentifier) == 0 {
			s.addResource(sets, vol, SetIdentifierNone)
		} else {
			found = false
			for _, tag := range vol.Tags {
				if s.matchTag(tag, setIdentifier) {
					s.addResource(sets, vol, *tag.Value)
					found = true
					break
				}
			}
			if !found {
				s.addResource(sets, vol, SetIdentifierNone)
			}
		}
	}
	return sets, nil
}

func (s *ec2Ops) Create(
	v *ec2.Volume,
	labels map[string]string,
) (*ec2.Volume, error) {

	req := &ec2.CreateVolumeInput{
		AvailabilityZone: v.AvailabilityZone,
		Encrypted:        v.Encrypted,
		KmsKeyId:         v.KmsKeyId,
		Size:             v.Size,
		VolumeType:       v.VolumeType,
		SnapshotId:       v.SnapshotId,
	}
	if *v.VolumeType == opsworks.VolumeTypeIo1 {
		req.Iops = v.Iops
	}

	newVol, err := s.ec2.CreateVolume(req)
	if err != nil {
		return nil, err
	}
	if err = s.waitStatus(
		*newVol.VolumeId,
		ec2.VolumeStateAvailable,
	); err != nil {
		return nil, s.rollbackCreate(*newVol.VolumeId, err)
	}
	if len(labels) > 0 {
		if err = s.ApplyTags(newVol, labels); err != nil {
			return nil, s.rollbackCreate(*newVol.VolumeId, err)
		}
	}

	return newVol, nil
}

func (s *ec2Ops) Delete(id string) error {
	req := &ec2.DeleteVolumeInput{VolumeId: &id}
	_, err := s.ec2.DeleteVolume(req)
	return err
}

func (s *ec2Ops) Attach(volumeID string) (string, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	devices, err := s.freeDevices()
	if err != nil {
		return "", err
	}
	req := &ec2.AttachVolumeInput{
		Device:     &devices[0],
		InstanceId: &s.instance,
		VolumeId:   &volumeID,
	}
	if _, err = s.ec2.AttachVolume(req); err != nil {
		return "", err
	}
	vol, err := s.waitAttachmentStatus(
		volumeID,
		ec2.VolumeAttachmentStateAttached,
		time.Minute,
	)
	if err != nil {
		return "", err
	}
	return s.DevicePath(vol)
}

func (s *ec2Ops) Detach(volumeID string) error {
	force := false
	req := &ec2.DetachVolumeInput{
		InstanceId: &s.instance,
		VolumeId:   &volumeID,
		Force:      &force,
	}
	if _, err := s.ec2.DetachVolume(req); err != nil {
		return err
	}
	_, err := s.waitAttachmentStatus(volumeID,
		ec2.VolumeAttachmentStateDetached,
		time.Minute,
	)
	return err
}

func (s *ec2Ops) Snapshot(
	volumeID string,
	readonly bool,
) (*ec2.Snapshot, error) {
	request := &ec2.CreateSnapshotInput{
		VolumeId: &volumeID,
	}
	return s.ec2.CreateSnapshot(request)
}

func (s *ec2Ops) DevicePath(vol *ec2.Volume) (string, error) {
	if vol.Attachments == nil || len(vol.Attachments) == 0 {
		return "", NewStorageError(ErrVolDetached,
			"Volume is detached", *vol.VolumeId)
	}
	if vol.Attachments[0].InstanceId == nil {
		return "", NewStorageError(ErrVolInval,
			"Unable to determine volume instance attachment", "")
	}
	if s.instance != *vol.Attachments[0].InstanceId {
		return "", NewStorageError(ErrVolAttachedOnRemoteNode,
			fmt.Sprintf("Volume attached on %q current instance %q",
				*vol.Attachments[0].InstanceId, s.instance),
			*vol.Attachments[0].InstanceId)

	}
	if vol.Attachments[0].State == nil {
		return "", NewStorageError(ErrVolInval,
			"Unable to determine volume attachment state", "")
	}
	if *vol.Attachments[0].State != ec2.VolumeAttachmentStateAttached {
		return "", NewStorageError(ErrVolInval,
			fmt.Sprintf("Invalid state %q, volume is not attached",
				*vol.Attachments[0].State), "")
	}
	if vol.Attachments[0].Device == nil {
		return "", NewStorageError(ErrVolInval,
			"Unable to determine volume attachment path", "")
	}
	dev := strings.TrimPrefix(*vol.Attachments[0].Device, "/dev/sd")
	if dev != *vol.Attachments[0].Device {
		dev = "/dev/xvd" + dev
	}
	return dev, nil
}
