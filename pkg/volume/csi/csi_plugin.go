/*
Copyright 2017 The Kubernetes Authors.

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

package csi

import (
	"errors"
	"fmt"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"context"

	"github.com/golang/glog"

	api "k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	csiapiinformer "k8s.io/csi-api/pkg/client/informers/externalversions"
	csiinformer "k8s.io/csi-api/pkg/client/informers/externalversions/csi/v1alpha1"
	csilister "k8s.io/csi-api/pkg/client/listers/csi/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"
)

const (
	csiPluginName = "kubernetes.io/csi"

	// TODO (vladimirvivien) implement a more dynamic way to discover
	// the unix domain socket path for each installed csi driver.
	// TODO (vladimirvivien) would be nice to name socket with a .sock extension
	// for consistency.
	csiAddrTemplate = "/var/lib/kubelet/plugins/%v/csi.sock"
	csiTimeout      = 15 * time.Second
	volNameSep      = "^"
	volDataFileName = "vol_data.json"
	fsTypeBlockName = "block"

	// TODO: increase to something useful
	csiResyncPeriod = time.Minute
)

type csiPlugin struct {
	host              volume.VolumeHost
	blockEnabled      bool
	csiDriverLister   csilister.CSIDriverLister
	csiDriverInformer csiinformer.CSIDriverInformer
}

// ProbeVolumePlugins returns implemented plugins
func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &csiPlugin{
		host:         nil,
		blockEnabled: utilfeature.DefaultFeatureGate.Enabled(features.CSIBlockVolume),
	}
	return []volume.VolumePlugin{p}
}

// volume.VolumePlugin methods
var _ volume.VolumePlugin = &csiPlugin{}

type csiDriver struct {
	driverName     string
	driverEndpoint string
}

type csiDriversStore struct {
	driversMap map[string]csiDriver
	sync.RWMutex
}

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct {
}

// TODO (verult) consider using a struct instead of global variables
// csiDrivers map keep track of all registered CSI drivers on the node and their
// corresponding sockets
var csiDrivers csiDriversStore

var nim nodeinfomanager.Interface

// PluginHandler is the plugin registration handler interface passed to the
// pluginwatcher module in kubelet
var PluginHandler = &RegistrationHandler{}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by CSI Driver registrar side car.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	glog.Infof(log("Trying to register a new plugin with name: %s endpoint: %s versions: %s",
		pluginName, endpoint, strings.Join(versions, ",")))

	return nil
}

// RegisterPlugin is called when a plugin can be registered
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string) error {
	glog.Infof(log("Register new plugin with name: %s at endpoint: %s", pluginName, endpoint))

	func() {
		// Storing endpoint of newly registered CSI driver into the map, where CSI driver name will be the key
		// all other CSI components will be able to get the actual socket of CSI drivers by its name.

		// It's not necessary to lock the entire RegistrationCallback() function because only the CSI
		// client depends on this driver map, and the CSI client does not depend on node information
		// updated in the rest of the function.
		csiDrivers.Lock()
		defer csiDrivers.Unlock()
		csiDrivers.driversMap[pluginName] = csiDriver{driverName: pluginName, driverEndpoint: endpoint}
	}()

	// Get node info from the driver.
	csi := newCsiDriverClient(pluginName)
	// TODO (verult) retry with exponential backoff, possibly added in csi client library.
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	driverNodeID, maxVolumePerNode, accessibleTopology, err := csi.NodeGetInfo(ctx)
	if err != nil {
		unregisterDriver(pluginName)
		return fmt.Errorf("error during CSI NodeGetInfo() call: %v", err)
	}

	err = nim.AddNodeInfo(pluginName, driverNodeID, maxVolumePerNode, accessibleTopology)
	if err != nil {
		unregisterDriver(pluginName)
		return fmt.Errorf("error updating CSI node info in the cluster: %v", err)
	}

	return nil
}

// DeRegisterPlugin is called when a plugin removed it's socket, signaling
// it is no longer available
// TODO: Handle DeRegistration
func (h *RegistrationHandler) DeRegisterPlugin(pluginName string) {
}

func (p *csiPlugin) Init(host volume.VolumeHost) error {
	p.host = host

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		csiClient := host.GetCSIClient()
		if csiClient == nil {
			glog.Warning("The client for CSI Custom Resources is not available, skipping informer initialization")
		} else {
			// Start informer for CSIDrivers.
			factory := csiapiinformer.NewSharedInformerFactory(csiClient, csiResyncPeriod)
			p.csiDriverInformer = factory.Csi().V1alpha1().CSIDrivers()
			p.csiDriverLister = p.csiDriverInformer.Lister()
			go factory.Start(wait.NeverStop)
		}
	}

	// Initializing csiDrivers map and label management channels
	csiDrivers = csiDriversStore{driversMap: map[string]csiDriver{}}
	nim = nodeinfomanager.NewNodeInfoManager(host.GetNodeName(), host)

	return nil
}

func (p *csiPlugin) GetPluginName() string {
	return csiPluginName
}

// GetvolumeName returns a concatenated string of CSIVolumeSource.Driver<volNameSe>CSIVolumeSource.VolumeHandle
// That string value is used in Detach() to extract driver name and volumeName.
func (p *csiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	csi, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("plugin.GetVolumeName failed to extract volume source from spec: %v", err))
		return "", err
	}

	// return driverName<separator>volumeHandle
	return fmt.Sprintf("%s%s%s", csi.Driver, volNameSep, csi.VolumeHandle), nil
}

func (p *csiPlugin) CanSupport(spec *volume.Spec) bool {
	// TODO (vladimirvivien) CanSupport should also take into account
	// the availability/registration of specified Driver in the volume source
	return spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil
}

func (p *csiPlugin) RequiresRemount() bool {
	return false
}

func (p *csiPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {
	pvSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		return nil, err
	}
	readOnly, err := getReadOnlyFromSpec(spec)
	if err != nil {
		return nil, err
	}

	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("failed to get a kubernetes client"))
		return nil, errors.New("failed to get a Kubernetes client")
	}

	csi := newCsiDriverClient(pvSource.Driver)

	mounter := &csiMountMgr{
		plugin:       p,
		k8s:          k8s,
		spec:         spec,
		pod:          pod,
		podUID:       pod.UID,
		driverName:   pvSource.Driver,
		volumeID:     pvSource.VolumeHandle,
		specVolumeID: spec.Name(),
		csiClient:    csi,
		readOnly:     readOnly,
	}

	// Save volume info in pod dir
	dir := mounter.GetPath()
	dataDir := path.Dir(dir) // dropoff /mount at end

	if err := os.MkdirAll(dataDir, 0750); err != nil {
		glog.Error(log("failed to create dir %#v:  %v", dataDir, err))
		return nil, err
	}
	glog.V(4).Info(log("created path successfully [%s]", dataDir))

	// persist volume info data for teardown
	node := string(p.host.GetNodeName())
	attachID := getAttachmentName(pvSource.VolumeHandle, pvSource.Driver, node)
	volData := map[string]string{
		volDataKey.specVolID:    spec.Name(),
		volDataKey.volHandle:    pvSource.VolumeHandle,
		volDataKey.driverName:   pvSource.Driver,
		volDataKey.nodeName:     node,
		volDataKey.attachmentID: attachID,
	}

	if err := saveVolumeData(dataDir, volDataFileName, volData); err != nil {
		glog.Error(log("failed to save volume info data: %v", err))
		if err := os.RemoveAll(dataDir); err != nil {
			glog.Error(log("failed to remove dir after error [%s]: %v", dataDir, err))
			return nil, err
		}
		return nil, err
	}

	glog.V(4).Info(log("mounter created successfully"))

	return mounter, nil
}

func (p *csiPlugin) NewUnmounter(specName string, podUID types.UID) (volume.Unmounter, error) {
	glog.V(4).Infof(log("setting up unmounter for [name=%v, podUID=%v]", specName, podUID))

	unmounter := &csiMountMgr{
		plugin:       p,
		podUID:       podUID,
		specVolumeID: specName,
	}

	// load volume info from file
	dir := unmounter.GetPath()
	dataDir := path.Dir(dir) // dropoff /mount at end
	data, err := loadVolumeData(dataDir, volDataFileName)
	if err != nil {
		glog.Error(log("unmounter failed to load volume data file [%s]: %v", dir, err))
		return nil, err
	}
	unmounter.driverName = data[volDataKey.driverName]
	unmounter.volumeID = data[volDataKey.volHandle]
	unmounter.csiClient = newCsiDriverClient(unmounter.driverName)

	return unmounter, nil
}

func (p *csiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glog.V(4).Info(log("plugin.ConstructVolumeSpec [pv.Name=%v, path=%v]", volumeName, mountPath))

	volData, err := loadVolumeData(mountPath, volDataFileName)
	if err != nil {
		glog.Error(log("plugin.ConstructVolumeSpec failed loading volume data using [%s]: %v", mountPath, err))
		return nil, err
	}

	glog.V(4).Info(log("plugin.ConstructVolumeSpec extracted [%#v]", volData))

	fsMode := api.PersistentVolumeFilesystem
	pv := &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: volData[volDataKey.specVolID],
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       volData[volDataKey.driverName],
					VolumeHandle: volData[volDataKey.volHandle],
				},
			},
			VolumeMode: &fsMode,
		},
	}

	return volume.NewSpecFromPersistentVolume(pv, false), nil
}

func (p *csiPlugin) SupportsMountOption() bool {
	// TODO (vladimirvivien) use CSI VolumeCapability.MountVolume.mount_flags
	// to probe for the result for this method
	return false
}

func (p *csiPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

// volume.AttachableVolumePlugin methods
var _ volume.AttachableVolumePlugin = &csiPlugin{}

var _ volume.DeviceMountableVolumePlugin = &csiPlugin{}

func (p *csiPlugin) NewAttacher() (volume.Attacher, error) {
	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("unable to get kubernetes client from host"))
		return nil, errors.New("unable to get Kubernetes client")
	}

	return &csiAttacher{
		plugin:        p,
		k8s:           k8s,
		waitSleepTime: 1 * time.Second,
	}, nil
}

func (p *csiPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return p.NewAttacher()
}

func (p *csiPlugin) NewDetacher() (volume.Detacher, error) {
	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("unable to get kubernetes client from host"))
		return nil, errors.New("unable to get Kubernetes client")
	}

	return &csiAttacher{
		plugin:        p,
		k8s:           k8s,
		waitSleepTime: 1 * time.Second,
	}, nil
}

func (p *csiPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return p.NewDetacher()
}

func (p *csiPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	m := p.host.GetMounter(p.GetPluginName())
	return m.GetMountRefs(deviceMountPath)
}

// BlockVolumePlugin methods
var _ volume.BlockVolumePlugin = &csiPlugin{}

func (p *csiPlugin) NewBlockVolumeMapper(spec *volume.Spec, podRef *api.Pod, opts volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	if !p.blockEnabled {
		return nil, errors.New("CSIBlockVolume feature not enabled")
	}

	pvSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		return nil, err
	}
	readOnly, err := getReadOnlyFromSpec(spec)
	if err != nil {
		return nil, err
	}

	glog.V(4).Info(log("setting up block mapper for [volume=%v,driver=%v]", pvSource.VolumeHandle, pvSource.Driver))
	client := newCsiDriverClient(pvSource.Driver)

	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("failed to get a kubernetes client"))
		return nil, errors.New("failed to get a Kubernetes client")
	}

	mapper := &csiBlockMapper{
		csiClient:  client,
		k8s:        k8s,
		plugin:     p,
		volumeID:   pvSource.VolumeHandle,
		driverName: pvSource.Driver,
		readOnly:   readOnly,
		spec:       spec,
		specName:   spec.Name(),
		podUID:     podRef.UID,
	}

	// Save volume info in pod dir
	dataDir := getVolumeDeviceDataDir(spec.Name(), p.host)

	if err := os.MkdirAll(dataDir, 0750); err != nil {
		glog.Error(log("failed to create data dir %s:  %v", dataDir, err))
		return nil, err
	}
	glog.V(4).Info(log("created path successfully [%s]", dataDir))

	// persist volume info data for teardown
	node := string(p.host.GetNodeName())
	attachID := getAttachmentName(pvSource.VolumeHandle, pvSource.Driver, node)
	volData := map[string]string{
		volDataKey.specVolID:    spec.Name(),
		volDataKey.volHandle:    pvSource.VolumeHandle,
		volDataKey.driverName:   pvSource.Driver,
		volDataKey.nodeName:     node,
		volDataKey.attachmentID: attachID,
	}

	if err := saveVolumeData(dataDir, volDataFileName, volData); err != nil {
		glog.Error(log("failed to save volume info data: %v", err))
		if err := os.RemoveAll(dataDir); err != nil {
			glog.Error(log("failed to remove dir after error [%s]: %v", dataDir, err))
			return nil, err
		}
		return nil, err
	}

	return mapper, nil
}

func (p *csiPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	if !p.blockEnabled {
		return nil, errors.New("CSIBlockVolume feature not enabled")
	}

	glog.V(4).Infof(log("setting up block unmapper for [Spec=%v, podUID=%v]", volName, podUID))
	unmapper := &csiBlockMapper{
		plugin:   p,
		podUID:   podUID,
		specName: volName,
	}

	// load volume info from file
	dataDir := getVolumeDeviceDataDir(unmapper.specName, p.host)
	data, err := loadVolumeData(dataDir, volDataFileName)
	if err != nil {
		glog.Error(log("unmapper failed to load volume data file [%s]: %v", dataDir, err))
		return nil, err
	}
	unmapper.driverName = data[volDataKey.driverName]
	unmapper.volumeID = data[volDataKey.volHandle]
	unmapper.csiClient = newCsiDriverClient(unmapper.driverName)

	return unmapper, nil
}

func (p *csiPlugin) ConstructBlockVolumeSpec(podUID types.UID, specVolName, mapPath string) (*volume.Spec, error) {
	if !p.blockEnabled {
		return nil, errors.New("CSIBlockVolume feature not enabled")
	}

	glog.V(4).Infof("plugin.ConstructBlockVolumeSpec [podUID=%s, specVolName=%s, path=%s]", string(podUID), specVolName, mapPath)

	dataDir := getVolumeDeviceDataDir(specVolName, p.host)
	volData, err := loadVolumeData(dataDir, volDataFileName)
	if err != nil {
		glog.Error(log("plugin.ConstructBlockVolumeSpec failed loading volume data using [%s]: %v", mapPath, err))
		return nil, err
	}

	glog.V(4).Info(log("plugin.ConstructBlockVolumeSpec extracted [%#v]", volData))

	blockMode := api.PersistentVolumeBlock
	pv := &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: volData[volDataKey.specVolID],
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       volData[volDataKey.driverName],
					VolumeHandle: volData[volDataKey.volHandle],
				},
			},
			VolumeMode: &blockMode,
		},
	}

	return volume.NewSpecFromPersistentVolume(pv, false), nil
}

func (p *csiPlugin) skipAttach(driver string) (bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		return false, nil
	}
	if p.csiDriverLister == nil {
		return false, errors.New("CSIDriver lister does not exist")
	}
	csiDriver, err := p.csiDriverLister.Get(driver)
	if err != nil {
		if apierrs.IsNotFound(err) {
			// Don't skip attach if CSIDriver does not exist
			return false, nil
		}
		return false, err
	}
	if csiDriver.Spec.AttachRequired != nil && *csiDriver.Spec.AttachRequired == false {
		return true, nil
	}
	return false, nil
}

func (p *csiPlugin) getPublishVolumeInfo(client clientset.Interface, handle, driver, nodeName string) (map[string]string, error) {
	skip, err := p.skipAttach(driver)
	if err != nil {
		return nil, err
	}
	if skip {
		return nil, nil
	}

	attachID := getAttachmentName(handle, driver, nodeName)

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	attachment, err := client.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		return nil, err // This err already has enough context ("VolumeAttachment xyz not found")
	}

	if attachment == nil {
		err = errors.New("no existing VolumeAttachment found")
		return nil, err
	}
	return attachment.Status.AttachmentMetadata, nil
}

func unregisterDriver(driverName string) {
	func() {
		csiDrivers.Lock()
		defer csiDrivers.Unlock()
		delete(csiDrivers.driversMap, driverName)
	}()

	if err := nim.RemoveNodeInfo(driverName); err != nil {
		glog.Errorf("Error unregistering CSI driver: %v", err)
	}
}
