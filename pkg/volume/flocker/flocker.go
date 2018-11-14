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

package flocker

import (
	"fmt"
	"os"
	"path"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/env"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"

	flockerapi "github.com/clusterhq/flocker-go"
	"k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&flockerPlugin{nil}}
}

type flockerPlugin struct {
	host volume.VolumeHost
}

type flockerVolume struct {
	volName string
	podUID  types.UID
	// dataset metadata name deprecated
	datasetName string
	// dataset uuid
	datasetUUID string
	//pod           *v1.Pod
	flockerClient flockerapi.Clientable
	manager       volumeManager
	plugin        *flockerPlugin
	mounter       mount.Interface
	volume.MetricsProvider
}

var _ volume.VolumePlugin = &flockerPlugin{}
var _ volume.PersistentVolumePlugin = &flockerPlugin{}
var _ volume.DeletableVolumePlugin = &flockerPlugin{}
var _ volume.ProvisionableVolumePlugin = &flockerPlugin{}

const (
	flockerPluginName = "kubernetes.io/flocker"

	defaultHost           = "localhost"
	defaultPort           = 4523
	defaultCACertFile     = "/etc/flocker/cluster.crt"
	defaultClientKeyFile  = "/etc/flocker/apiuser.key"
	defaultClientCertFile = "/etc/flocker/apiuser.crt"
	defaultMountPath      = "/flocker"

	timeoutWaitingForVolume = 2 * time.Minute
	tickerWaitingForVolume  = 5 * time.Second
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(flockerPluginName), volName)
}

func makeGlobalFlockerPath(datasetUUID string) string {
	return path.Join(defaultMountPath, datasetUUID)
}

func (p *flockerPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	return nil
}

func (p *flockerPlugin) GetPluginName() string {
	return flockerPluginName
}

func (p *flockerPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.DatasetName, nil
}

func (p *flockerPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Flocker != nil) ||
		(spec.Volume != nil && spec.Volume.Flocker != nil)
}

func (p *flockerPlugin) RequiresRemount() bool {
	return false
}

func (p *flockerPlugin) SupportsMountOption() bool {
	return false
}

func (p *flockerPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (p *flockerPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

func (p *flockerPlugin) getFlockerVolumeSource(spec *volume.Spec) (*v1.FlockerVolumeSource, bool) {
	// AFAIK this will always be r/w, but perhaps for the future it will be needed
	readOnly := false

	if spec.Volume != nil && spec.Volume.Flocker != nil {
		return spec.Volume.Flocker, readOnly
	}
	return spec.PersistentVolume.Spec.Flocker, readOnly
}

func (p *flockerPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return p.newMounterInternal(spec, pod.UID, &flockerUtil{}, p.host.GetMounter(p.GetPluginName()))
}

func (p *flockerPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager volumeManager, mounter mount.Interface) (volume.Mounter, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	datasetName := volumeSource.DatasetName
	datasetUUID := volumeSource.DatasetUUID

	return &flockerVolumeMounter{
		flockerVolume: &flockerVolume{
			podUID:          podUID,
			volName:         spec.Name(),
			datasetName:     datasetName,
			datasetUUID:     datasetUUID,
			mounter:         mounter,
			manager:         manager,
			plugin:          p,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), p.host)),
		},
		readOnly: readOnly}, nil
}

func (p *flockerPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return p.newUnmounterInternal(volName, podUID, &flockerUtil{}, p.host.GetMounter(p.GetPluginName()))
}

func (p *flockerPlugin) newUnmounterInternal(volName string, podUID types.UID, manager volumeManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &flockerVolumeUnmounter{&flockerVolume{
		podUID:          podUID,
		volName:         volName,
		manager:         manager,
		mounter:         mounter,
		plugin:          p,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, p.host)),
	}}, nil
}

func (p *flockerPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	flockerVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Flocker: &v1.FlockerVolumeSource{
				DatasetName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(flockerVolume), nil
}

func (b *flockerVolume) GetDatasetUUID() (datasetUUID string, err error) {

	// return UUID if set
	if len(b.datasetUUID) > 0 {
		return b.datasetUUID, nil
	}

	if b.flockerClient == nil {
		return "", fmt.Errorf("Flocker client is not initialized")
	}

	// lookup in flocker API otherwise
	return b.flockerClient.GetDatasetID(b.datasetName)
}

type flockerVolumeMounter struct {
	*flockerVolume
	readOnly bool
}

func (b *flockerVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *flockerVolumeMounter) CanMount() error {
	return nil
}

func (b *flockerVolumeMounter) GetPath() string {
	return getPath(b.podUID, b.volName, b.plugin.host)
}

// SetUp bind mounts the disk global mount to the volume path.
func (b *flockerVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// newFlockerClient uses environment variables and pod attributes to return a
// flocker client capable of talking with the Flocker control service.
func (p *flockerPlugin) newFlockerClient(hostIP string) (*flockerapi.Client, error) {
	host := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_HOST", defaultHost)
	port, err := env.GetEnvAsIntOrFallback("FLOCKER_CONTROL_SERVICE_PORT", defaultPort)

	if err != nil {
		return nil, err
	}
	caCertPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CA_FILE", defaultCACertFile)
	keyPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CLIENT_KEY_FILE", defaultClientKeyFile)
	certPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CLIENT_CERT_FILE", defaultClientCertFile)

	c, err := flockerapi.NewClient(host, port, hostIP, caCertPath, keyPath, certPath)
	return c, err
}

func (b *flockerVolumeMounter) newFlockerClient() (*flockerapi.Client, error) {

	hostIP, err := b.plugin.host.GetHostIP()
	if err != nil {
		return nil, err
	}

	return b.plugin.newFlockerClient(hostIP.String())
}

/*
SetUpAt will setup a Flocker volume following this flow of calls to the Flocker
control service:

1. Get the dataset id for the given volume name/dir
2. It should already be there, if it's not the user needs to manually create it
3. Check the current Primary UUID
4. If it doesn't match with the Primary UUID that we got on 2, then we will
   need to update the Primary UUID for this volume.
5. Wait until the Primary UUID was updated or timeout.
*/
func (b *flockerVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	var err error
	if b.flockerClient == nil {
		b.flockerClient, err = b.newFlockerClient()
		if err != nil {
			return err
		}
	}

	datasetUUID, err := b.GetDatasetUUID()
	if err != nil {
		return fmt.Errorf("The datasetUUID for volume with datasetName='%s' can not be found using flocker: %s", b.datasetName, err)
	}

	datasetState, err := b.flockerClient.GetDatasetState(datasetUUID)
	if err != nil {
		return fmt.Errorf("The datasetState for volume with datasetUUID='%s' could not determinted uusing flocker: %s", datasetUUID, err)
	}

	primaryUUID, err := b.flockerClient.GetPrimaryUUID()
	if err != nil {
		return err
	}

	if datasetState.Primary != primaryUUID {
		if err := b.updateDatasetPrimary(datasetUUID, primaryUUID); err != nil {
			return err
		}
		_, err := b.flockerClient.GetDatasetState(datasetUUID)
		if err != nil {
			return fmt.Errorf("The volume with datasetUUID='%s' migrated unsuccessfully", datasetUUID)
		}
	}

	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	klog.V(4).Infof("flockerVolume set up: %s %v %v, datasetUUID %v readOnly %v", dir, !notMnt, err, datasetUUID, b.readOnly)
	if err != nil && !os.IsNotExist(err) {
		klog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		klog.Errorf("mkdir failed on disk %s (%v)", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	globalFlockerPath := makeGlobalFlockerPath(datasetUUID)
	klog.V(4).Infof("attempting to mount %s", dir)

	err = b.mounter.Mount(globalFlockerPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			klog.Errorf("isLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				klog.Errorf("failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				klog.Errorf("isLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				klog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		klog.Errorf("mount of disk %s failed: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}

	klog.V(4).Infof("successfully mounted %s", dir)
	return nil
}

// updateDatasetPrimary will update the primary in Flocker and wait for it to
// be ready. If it never gets to ready state it will timeout and error.
func (b *flockerVolumeMounter) updateDatasetPrimary(datasetUUID string, primaryUUID string) error {
	// We need to update the primary and wait for it to be ready
	_, err := b.flockerClient.UpdatePrimaryForDataset(primaryUUID, datasetUUID)
	if err != nil {
		return err
	}

	timeoutChan := time.NewTimer(timeoutWaitingForVolume)
	defer timeoutChan.Stop()
	tickChan := time.NewTicker(tickerWaitingForVolume)
	defer tickChan.Stop()

	for {
		if s, err := b.flockerClient.GetDatasetState(datasetUUID); err == nil && s.Primary == primaryUUID {
			return nil
		}

		select {
		case <-timeoutChan.C:
			return fmt.Errorf(
				"Timed out waiting for the datasetUUID: '%s' to be moved to the primary: '%s'\n%v",
				datasetUUID, primaryUUID, err,
			)
		case <-tickChan.C:
			break
		}
	}

}

func getVolumeSource(spec *volume.Spec) (*v1.FlockerVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Flocker != nil {
		return spec.Volume.Flocker, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Flocker != nil {
		return spec.PersistentVolume.Spec.Flocker, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Flocker volume type")
}

type flockerVolumeUnmounter struct {
	*flockerVolume
}

var _ volume.Unmounter = &flockerVolumeUnmounter{}

func (c *flockerVolumeUnmounter) GetPath() string {
	return getPath(c.podUID, c.volName, c.plugin.host)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *flockerVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt unmounts the bind mount
func (c *flockerVolumeUnmounter) TearDownAt(dir string) error {
	return util.UnmountPath(dir, c.mounter)
}

func (p *flockerPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return p.newDeleterInternal(spec, &flockerUtil{})
}

func (p *flockerPlugin) newDeleterInternal(spec *volume.Spec, manager volumeManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Flocker == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Flocker is nil")
	}
	return &flockerVolumeDeleter{
		flockerVolume: &flockerVolume{
			volName:     spec.Name(),
			datasetName: spec.PersistentVolume.Spec.Flocker.DatasetName,
			datasetUUID: spec.PersistentVolume.Spec.Flocker.DatasetUUID,
			manager:     manager,
		}}, nil
}

func (p *flockerPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return p.newProvisionerInternal(options, &flockerUtil{})
}

func (p *flockerPlugin) newProvisionerInternal(options volume.VolumeOptions, manager volumeManager) (volume.Provisioner, error) {
	return &flockerVolumeProvisioner{
		flockerVolume: &flockerVolume{
			manager: manager,
			plugin:  p,
		},
		options: options,
	}, nil
}
