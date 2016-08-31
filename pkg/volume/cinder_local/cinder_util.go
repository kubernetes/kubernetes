/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cinder_local

import (
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/extensions/volumeactions"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

const savedSecretFilename = "secret.json"

type cinderVolumeHandler interface {
	// AttachDisk attaches the volume to the host and returns the devicePath
	AttachDisk(connInfo connectionInfo) (string, error)
	// DetachDisk detaches the volume from the host
	DetachDisk(connInfo connectionInfo) error
}

var volTypeHandlers = map[string]cinderVolumeHandler{
	"rbd": &rbdHandler{},
}

type connectionInfo struct {
	DriverVolumeType string `json:"driver_volume_type"`
	Data             struct {
		Name string `json:"name"`
	} `json:"data"`
}

type cancelableFunc struct {
	fn       func()
	canceled bool
}

func cancelable(f func()) cancelableFunc {
	return cancelableFunc{
		fn: f,
	}
}

func (c *cancelableFunc) call() {
	if !c.canceled {
		c.fn()
	}
}

func (c *cancelableFunc) cancel() {
	c.canceled = true
}

type cinderDiskUtil struct {
}

func getSecret(host volume.VolumeHost, secretRef string) (*api.Secret, error) {
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("cannot get kube client")
	}

	ns, secretName := splitSecretRef(secretRef)

	secret, err := kubeClient.Core().Secrets(ns).Get(secretName)
	if err != nil {
		return nil, fmt.Errorf("could not get secret %s/%s: %v", ns, secretRef, err)
	}

	return secret, nil
}

func authOptionsFromSecret(secret *api.Secret) (gophercloud.AuthOptions, error) {
	nilOptions := gophercloud.AuthOptions{}

	authURL, ok := secret.Data["authURL"]
	if !ok {
		return nilOptions, fmt.Errorf("secret missing 'authURL' key")
	}

	username, ok := secret.Data["username"]
	if !ok {
		return nilOptions, fmt.Errorf("secret missing 'username' key")
	}

	password, ok := secret.Data["password"]
	if !ok {
		return nilOptions, fmt.Errorf("secret missing 'password' key")
	}

	project := secret.Data["project"]
	domain := secret.Data["domain"]

	return gophercloud.AuthOptions{
		IdentityEndpoint: string(authURL),
		Username:         string(username),
		Password:         string(password),
		TenantName:       string(project),
		DomainName:       string(domain),
	}, nil
}

func secretFullname(path string) string {
	return filepath.Join(path, savedSecretFilename)
}

func persistSecret(path string, secret *api.Secret) error {
	scheme := runtime.NewScheme()
	s := json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, true)

	if err := os.MkdirAll(path, 0750); err != nil {
		return fmt.Errorf("failed to persist secret: %v", err)
	}

	f, err := os.Create(secretFullname(path))
	if err != nil {
		return fmt.Errorf("failed to create file to persist secret: %v", err)
	}
	defer f.Close()

	if err := s.Encode(secret, f); err != nil {
		return fmt.Errorf("failed to persist secret to JSON: %v", err)
	}

	return nil
}

func loadSecret(path string) (*api.Secret, error) {
	scheme := runtime.NewScheme()
	s := json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, true)

	contents, err := ioutil.ReadFile(secretFullname(path))
	if err != nil {
		return nil, fmt.Errorf("failed to read persisted secret file: %v", err)
	}

	secret := &api.Secret{}
	_, _, err = s.Decode(contents, nil, secret)
	if err != nil {
		return nil, fmt.Errorf("failed to decode persisted secret file: %v", err)
	}

	return secret, nil
}

func newCinderClient(host volume.VolumeHost, secret *api.Secret) (*gophercloud.ServiceClient, error) {
	authOpts, err := authOptionsFromSecret(secret)
	if err != nil {
		return nil, err
	}

	provider, err := openstack.AuthenticatedClient(authOpts)
	if err != nil {
		glog.V(4).Infof("AuthenticatedClient failed: %v", err)
		return nil, err
	}

	eo := gophercloud.EndpointOpts{}

	client, err := openstack.NewBlockStorageV2(provider, eo)
	if err != nil {
		glog.V(4).Infof("NewBlockStorageV2 failed: %v", err)
		return nil, err
	}

	return client, nil
}

func initializeConnection(client *gophercloud.ServiceClient, volumeID string, host volume.VolumeHost) (connectionInfo, error) {
	ip, err := host.GetHostIP()
	if err != nil {
		return connectionInfo{}, fmt.Errorf("could not get host IP: %v", err)
	}

	connOpts := volumeactions.InitializeConnectionOpts{
		IP:       ip.String(),
		Host:     host.GetHostName(),
		Platform: build.Default.GOARCH,
		OSType:   build.Default.GOOS,
	}

	icResult := volumeactions.InitializeConnection(client, volumeID, &connOpts)
	if icResult.Err != nil {
		return connectionInfo{}, fmt.Errorf("cinder 'initialize_connection' failed: %v", icResult.Err)
	}

	var s struct {
		ConnInfo connectionInfo `json:"connection_info"`
	}
	if err := icResult.ExtractInto(&s); err != nil {
		return connectionInfo{}, fmt.Errorf("cinder initialize_connection returned bad result: %v", err)
	}

	return s.ConnInfo, nil
}

func finalizeAttach(client *gophercloud.ServiceClient, volumeID string, readOnly bool, mntPoint string, host volume.VolumeHost) error {
	attachMode := volumeactions.ReadWrite
	if readOnly {
		// TODO: it's unclear if the block device should be mounted read-only
		// as some filesystems will still want to write to it even when mounted
		// in ro mode
		attachMode = volumeactions.ReadOnly
	}

	attachOpts := volumeactions.AttachOpts{
		MountPoint: mntPoint,
		HostName:   host.GetHostName(),
		Mode:       attachMode,
	}

	result := volumeactions.Attach(client, volumeID, &attachOpts)
	return result.Err
}

func finalizeDetach(client *gophercloud.ServiceClient, volumeID string, host volume.VolumeHost) error {
	ip, err := host.GetHostIP()
	if err != nil {
		return fmt.Errorf("could not get host IP: %v", err)
	}

	connOpts := volumeactions.TerminateConnectionOpts{
		IP:       ip.String(),
		Host:     host.GetHostName(),
		Platform: build.Default.GOARCH,
		OSType:   build.Default.GOOS,
	}

	// It's ok for rbd but not sure if this call is in general idempotent
	tcResult := volumeactions.TerminateConnection(client, volumeID, &connOpts)
	if tcResult.Err != nil {
		return fmt.Errorf("cinder 'terminate_connection' failed: %v", tcResult.Err)
	}

	// Also not sure if it's idempotent
	detachOpts := volumeactions.DetachOpts{}
	detachResult := volumeactions.Detach(client, volumeID, &detachOpts)
	if detachResult.Err != nil {
		return fmt.Errorf("cinder 'detach' failed: %v", detachResult.Err)
	}

	return nil
}

func mountFS(mounter mount.Interface, devicePath string, fsType string, readOnly bool, mntPoint string) error {
	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}

	notmnt, err := mounter.IsLikelyNotMountPoint(mntPoint)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(mntPoint, 0750); err != nil {
				return fmt.Errorf("error creating %s: %v", mntPoint, err)
			}
			notmnt = true
		} else {
			return fmt.Errorf("could not determine if %s is a mountpoint: %v", mntPoint, err)
		}
	}
	if notmnt {
		blockDeviceMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = blockDeviceMounter.FormatAndMount(devicePath, mntPoint, fsType, options)
		if err != nil {
			return fmt.Errorf("error mounting %s to %s: %v", devicePath, mntPoint, err)
		}
		glog.V(4).Infof("Safe mount successful: %q", devicePath)
	}

	return nil
}

// Attaches a disk specified by a volume.CinderPersistenDisk to the current kubelet.
// Mounts the disk to it's global path.
func (util *cinderDiskUtil) AttachDisk(m *cinderVolumeMounter, mntPoint string) error {
	glog.V(4).Infof("cinderDiskUtil.AttachDisk: %v", mntPoint)

	secret, err := getSecret(m.plugin.host, m.source.SecretRef)
	if err != nil {
		return err
	}

	client, err := newCinderClient(m.plugin.host, secret)
	if err != nil {
		return err
	}

	// By detach time, secret might be deleted, persist it into the
	// mount point directory. The volume will get mounted over it
	// hiding it from plain view.
	if err = persistSecret(mntPoint, secret); err != nil {
		return err
	}

	glog.V(4).Infof("Calling volumeactions.Reserve(%v)", m.source.VolumeID)
	rsvResult := volumeactions.Reserve(client, m.source.VolumeID)
	if rsvResult.Err != nil {
		return fmt.Errorf("cinder 'reserve' failed: %v", rsvResult.Err)
	}

	// If we fail along the way, rollback the reservation
	unreserve := cancelable(func() {
		volumeactions.Unreserve(client, m.source.VolumeID)
	})
	defer unreserve.call()

	connInfo, err := initializeConnection(client, m.source.VolumeID, m.plugin.host)
	if err != nil {
		return err
	}

	volType := strings.ToLower(connInfo.DriverVolumeType)

	volHandler, ok := volTypeHandlers[volType]
	if !ok {
		return fmt.Errorf("%q volume type is not supported", volType)
	}

	glog.V(4).Infof("cinderDiskUtil.AttachDisk: calling volHandler.AttachDisk")
	devicePath, err := volHandler.AttachDisk(connInfo)
	if err != nil {
		return err
	}
	glog.V(4).Infof("cinderDiskUtil.AttachDisk: volHandler.AttachDisk succeeded, calling mountFS")

	err = mountFS(m.mounter, devicePath, m.source.FSType, m.readOnly, mntPoint)
	if err != nil {
		return err
	}

	glog.V(4).Infof("cinderDiskUtil.AttachDisk: calling finalizeAttach")
	if err = finalizeAttach(client, m.source.VolumeID, m.readOnly, mntPoint, m.plugin.host); err == nil {
		unreserve.cancel()
	}

	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *cinderDiskUtil) DetachDisk(u *cinderVolumeUnmounter, mntPoint string) error {
	if err := u.mounter.Unmount(mntPoint); err != nil {
		// ignore error if it's been already unmounted
		notmnt, err2 := u.mounter.IsLikelyNotMountPoint(mntPoint)
		if err2 != nil || !notmnt {
			return err
		}
	}

	glog.V(2).Infof("Successfully unmounted main device: %s\n", mntPoint)

	secret, err := loadSecret(mntPoint)
	if err != nil {
		return err
	}

	client, err := newCinderClient(u.plugin.host, secret)
	if err != nil {
		return err
	}

	// Need to call os-initialize_connection to get the volume type
	connInfo, err := initializeConnection(client, u.volumeID, u.plugin.host)
	if err != nil {
		return err
	}

	// Technically required but skip for two reasons:
	// 1. We need to do the unmount before contacting cinder API in order to get the secret
	// 2. If something goes wrong, we could reset the detaching state back to in-use but that could fail as well
	//bdResult := volumeactions.BeginDetaching(client, volumeID)
	//if bdResult.Err != nil {
	//	return nil, err
	//}

	volHandler, ok := volTypeHandlers[connInfo.DriverVolumeType]
	if !ok {
		return fmt.Errorf("%v volume type is not supported", connInfo.DriverVolumeType)
	}

	if err := volHandler.DetachDisk(connInfo); err != nil {
		return err
	}

	if err = finalizeDetach(client, u.volumeID, u.plugin.host); err != nil {
		return err
	}

	if err := os.RemoveAll(mntPoint); err != nil {
		return fmt.Errorf("could not remove %s: %v", mntPoint, err)
	}

	glog.V(2).Infof("Successfully detached cinder-local volume %s", u.volumeID)
	return nil
}

func (util *cinderDiskUtil) DeleteVolume(d *cinderVolumeDeleter) error {
	secret, err := getSecret(d.plugin.host, d.pv.Spec.CinderLocal.SecretRef)
	if err != nil {
		return err
	}

	client, err := newCinderClient(d.plugin.host, secret)
	if err != nil {
		return err
	}

	res := volumes.Delete(client, d.pv.Spec.PersistentVolumeSource.CinderLocal.VolumeID)
	if res.Err != nil {
		return fmt.Errorf("cinder 'delete' failed: %v", res.Err)
	}

	glog.V(2).Infof("Successfully deleted cinder volume %s", d.pv.Spec.CinderLocal.VolumeID)
	return nil
}

func splitSecretRef(secretRef string) (string, string) {
	parts := strings.SplitN(secretRef, "/", 2)
	if len(parts) == 1 {
		return "default", parts[0]
	} else {
		return parts[0], parts[1]
	}
}

func (util *cinderDiskUtil) CreateVolume(p *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef string, err error) {
	volSizeBytes := p.options.Capacity.Value()
	// Cinder works with gigabytes, convert to GiB with rounding up
	volSizeGB := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	name := volume.GenerateVolumeName(p.options.ClusterName, p.options.PVName, 255) // Cinder volume name can have up to 255 characters

	var md map[string]string
	if p.options.CloudTags != nil {
		md = *p.options.CloudTags
	}

	vtype := ""
	availability := ""
	// Apply ProvisionerParameters (case-insensitive)
	for k, v := range p.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			vtype = v
		case "availability":
			availability = v
		case "secretref":
			secretRef = v
		default:
			return "", 0, "", fmt.Errorf("invalid option %q for volume plugin %s", k, p.plugin.GetPluginName())
		}
	}

	if secretRef == "" {
		return "", 0, "", fmt.Errorf("StorageClass missing 'secretRef' in parameters")
	}

	// TODO: implement p.options.ProvisionerSelector parsing
	if p.options.Selector != nil {
		return "", 0, "", fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on CinderLocal")
	}

	secret, err := getSecret(p.plugin.host, secretRef)
	if err != nil {
		return "", 0, "", err
	}

	client, err := newCinderClient(p.plugin.host, secret)
	if err != nil {
		return "", 0, "", err
	}

	createOpts := volumes.CreateOpts{
		Size:             volSizeGB,
		Name:             p.options.PVName,
		Metadata:         md,
		AvailabilityZone: availability,
		VolumeType:       vtype,
	}

	res := volumes.Create(client, &createOpts)
	if res.Err != nil {
		return "", 0, "", fmt.Errorf("cinder 'create' failed: %v", res.Err)
	}

	vol, err := res.Extract()
	if err != nil {
		return "", 0, "", fmt.Errorf("cinder volume creation returned bad result: %v", err)
	}

	glog.V(2).Infof("Successfully created cinder volume %q (%v)", name, vol.ID)
	return vol.ID, volSizeGB, secretRef, nil
}
