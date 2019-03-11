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
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"k8s.io/klog"

	api "k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	csiapiinformer "k8s.io/client-go/informers"
	csiinformer "k8s.io/client-go/informers/storage/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	csilister "k8s.io/client-go/listers/storage/v1beta1"
	csitranslationplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"
)

const (
	// CSIPluginName is the name of the in-tree CSI Plugin
	CSIPluginName = "kubernetes.io/csi"

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

var deprecatedSocketDirVersions = []string{"0.1.0", "0.2.0", "0.3.0", "0.4.0"}

type csiPlugin struct {
	host              volume.VolumeHost
	blockEnabled      bool
	csiDriverLister   csilister.CSIDriverLister
	csiDriverInformer csiinformer.CSIDriverInformer

	drivers *DriversStore
	nim     nodeinfomanager.Interface
}

//TODO (vladimirvivien) add this type to storage api
type driverMode string

const persistentDriverMode driverMode = "persistent"
const ephemeralDriverMode driverMode = "ephemeral"

// PluginHandler is only needed for the kubelet to add the CSI plugin as a pluginwatcher handler
// as soon as the kubelet discovers the CSI plugin via name and not via this
// package-global variable, we can remove that singleton again.
// TODO(hoergaarden) remove when kubelet changes to get volume plugin by name are merged
var PluginHandler *csiPlugin
var once sync.Once

// ProbeVolumePlugins returns implemented plugins
func ProbeVolumePlugins() []volume.VolumePlugin {
	// TODO(hoergaarden) remove when kubelet changes to get volume plugin by name are merged
	once.Do(func() {
		PluginHandler = &csiPlugin{
			host:         nil,
			blockEnabled: utilfeature.DefaultFeatureGate.Enabled(features.CSIBlockVolume),
			drivers:      &DriversStore{},
		}
	})
	return []volume.VolumePlugin{PluginHandler}
}

var _ pluginwatcher.PluginHandler = &csiPlugin{}

// volume.VolumePlugin methods
var _ volume.VolumePlugin = &csiPlugin{}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by CSI Driver registrar side car.
// implements: PluginHandler
func (p *csiPlugin) ValidatePlugin(pluginName string, endpoint string, versions []string, foundInDeprecatedDir bool) error {
	klog.Infof(log("Trying to validate a new CSI Driver with name: %s endpoint: %s versions: %s, foundInDeprecatedDir: %v",
		pluginName, endpoint, strings.Join(versions, ","), foundInDeprecatedDir))

	if foundInDeprecatedDir {
		// CSI 0.x drivers used /var/lib/kubelet/plugins as the socket dir.
		// This was deprecated as the socket dir for kubelet drivers, in lieu of a dedicated dir /var/lib/kubelet/plugins_registry
		// The deprecated dir will only be allowed for a whitelisted set of old versions.
		// CSI 1.x drivers should use the /var/lib/kubelet/plugins_registry
		if !isDeprecatedSocketDirAllowed(versions) {
			err := fmt.Errorf("socket for CSI driver %q versions %v was found in a deprecated dir. Drivers implementing CSI 1.x+ must use the new dir", pluginName, versions)
			klog.Error(err)
			return err
		}
	}

	_, err := p.validateVersions("ValidatePlugin", pluginName, endpoint, versions)
	return err
}

// RegisterPlugin is called when a plugin can be registered
// implements: PluginHandler
func (p *csiPlugin) RegisterPlugin(pluginName string, endpoint string, versions []string) error {
	klog.Infof(log("Register new plugin with name: %s at endpoint: %s", pluginName, endpoint))

	highestSupportedVersion, err := p.validateVersions("RegisterPlugin", pluginName, endpoint, versions)
	if err != nil {
		return err
	}

	// Storing endpoint of newly registered CSI driver into the map, where CSI driver name will be the key
	// all other CSI components will be able to get the actual socket of CSI drivers by its name.
	p.drivers.Set(pluginName, Driver{
		endpoint:                endpoint,
		highestSupportedVersion: highestSupportedVersion,
	})

	// Get node info from the driver.
	csi, err := p.newCsiDriverClient(csiDriverName(pluginName))
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	driverNodeID, maxVolumePerNode, accessibleTopology, err := csi.NodeGetInfo(ctx)
	if err != nil {
		if unregErr := p.unregisterDriver(pluginName); unregErr != nil {
			klog.Error(log("RegisterPlugin failed to unregister plugin due to previous error: %v", unregErr))
		}
		return err
	}

	err = p.nim.InstallCSIDriver(pluginName, driverNodeID, maxVolumePerNode, accessibleTopology)
	if err != nil {
		if unregErr := p.unregisterDriver(pluginName); unregErr != nil {
			klog.Error(log("RegisterPlugin failed to unregister plugin due to previous error: %v", unregErr))
		}
		return err
	}

	return nil
}

// for PluginHandler
func (p *csiPlugin) validateVersions(callerName, pluginName string, endpoint string, versions []string) (*utilversion.Version, error) {
	if len(versions) == 0 {
		err := fmt.Errorf("%s for CSI driver %q failed. Plugin returned an empty list for supported versions", callerName, pluginName)
		klog.Error(err)
		return nil, err
	}

	// Validate version
	newDriverHighestVersion, err := highestSupportedVersion(versions)
	if err != nil {
		err := fmt.Errorf("%s for CSI driver %q failed. None of the versions specified %q are supported. err=%v", callerName, pluginName, versions, err)
		klog.Error(err)
		return nil, err
	}

	existingDriver, driverExists := p.drivers.Get(pluginName)
	if driverExists {
		if !existingDriver.highestSupportedVersion.LessThan(newDriverHighestVersion) {
			err := fmt.Errorf("%s for CSI driver %q failed. Another driver with the same name is already registered with a higher supported version: %q", callerName, pluginName, existingDriver.highestSupportedVersion)
			klog.Error(err)
			return nil, err
		}
	}

	return newDriverHighestVersion, nil
}

func (p *csiPlugin) newCsiDriverClient(driverName csiDriverName) (*csiDriverClient, error) {
	if driverName == "" {
		return nil, fmt.Errorf("driver name is empty")
	}

	addr := fmt.Sprintf(csiAddrTemplate, driverName)
	requiresV0Client := true
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPluginsWatcher) {
		existingDriver, driverExists := p.drivers.Get(string(driverName))
		if !driverExists {
			return nil, fmt.Errorf("driver name %s not found in the list of registered CSI drivers", driverName)
		}

		addr = existingDriver.endpoint
		requiresV0Client = versionRequiresV0Client(existingDriver.highestSupportedVersion)
	}

	nodeV1ClientCreator := newV1NodeClient
	nodeV0ClientCreator := newV0NodeClient
	if requiresV0Client {
		nodeV1ClientCreator = nil
	} else {
		nodeV0ClientCreator = nil
	}

	return &csiDriverClient{
		driverName:          driverName,
		addr:                csiAddr(addr),
		nodeV1ClientCreator: nodeV1ClientCreator,
		nodeV0ClientCreator: nodeV0ClientCreator,
	}, nil
}

// DeRegisterPlugin is called when a plugin removed its socket, signaling
// it is no longer available
// implements: PluginHandler
func (p *csiPlugin) DeRegisterPlugin(pluginName string) {
	klog.V(4).Info(log("registrationHandler.DeRegisterPlugin request for plugin %s", pluginName))
	if err := p.unregisterDriver(pluginName); err != nil {
		klog.Error(log("registrationHandler.DeRegisterPlugin failed: %v", err))
	}
}

// Init initializes the plugin.
// This must be called exactly once before any New* (NewMounter, NewAttacher,
// ...) methods are called the plugin.
// implements: VolumePlugin
func (p *csiPlugin) Init(host volume.VolumeHost) error {
	p.host = host

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		csiClient := host.GetKubeClient()
		if csiClient == nil {
			klog.Warning(log("kubeclient not set, assuming standalone kubelet"))
		} else {
			// Start informer for CSIDrivers.
			factory := csiapiinformer.NewSharedInformerFactory(csiClient, csiResyncPeriod)
			p.csiDriverInformer = factory.Storage().V1beta1().CSIDrivers()
			p.csiDriverLister = p.csiDriverInformer.Lister()
			go factory.Start(wait.NeverStop)
		}
	}

	var migratedPlugins = map[string](func() bool){
		csitranslationplugins.GCEPDInTreePluginName: func() bool {
			return utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) && utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationGCE)
		},
		csitranslationplugins.AWSEBSInTreePluginName: func() bool {
			return utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) && utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationAWS)
		},
		csitranslationplugins.CinderInTreePluginName: func() bool {
			return utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) && utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationOpenStack)
		},
	}

	// Initializing the label management channels
	p.nim = nodeinfomanager.NewNodeInfoManager(host.GetNodeName(), host, migratedPlugins)

	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) &&
		utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) {
		// This function prevents Kubelet from posting Ready status until CSINodeInfo
		// is both installed and initialized
		if err := initializeCSINode(host, p.nim); err != nil {
			return fmt.Errorf("failed to initialize CSINodeInfo: %v", err)
		}
	}

	return nil
}

func initializeCSINode(host volume.VolumeHost, nim nodeinfomanager.Interface) error {
	kvh, ok := host.(volume.KubeletVolumeHost)
	if !ok {
		klog.V(4).Info("Cast from VolumeHost to KubeletVolumeHost failed. Skipping CSINodeInfo initialization, not running on kubelet")
		return nil
	}
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		// Kubelet running in standalone mode. Skip CSINodeInfo initialization
		klog.Warning("Skipping CSINodeInfo initialization, kubelet running in standalone mode")
		return nil
	}

	kvh.SetKubeletError(errors.New("CSINodeInfo is not yet initialized"))

	go func() {
		defer utilruntime.HandleCrash()

		// Backoff parameters tuned to retry over 140 seconds. Will fail and restart the Kubelet
		// after max retry steps.
		initBackoff := wait.Backoff{
			Steps:    6,
			Duration: 15 * time.Millisecond,
			Factor:   6.0,
			Jitter:   0.1,
		}
		err := wait.ExponentialBackoff(initBackoff, func() (bool, error) {
			klog.V(4).Infof("Initializing migrated drivers on CSINodeInfo")
			err := nim.InitializeCSINodeWithAnnotation()
			if err != nil {
				kvh.SetKubeletError(fmt.Errorf("Failed to initialize CSINodeInfo: %v", err))
				klog.Errorf("Failed to initialize CSINodeInfo: %v", err)
				return false, nil
			}

			// Successfully initialized drivers, allow Kubelet to post Ready
			kvh.SetKubeletError(nil)
			return true, nil
		})
		if err != nil {
			// 2 releases after CSIMigration and all CSIMigrationX (where X is a volume plugin)
			// are permanently enabled the apiserver/controllers can assume that the kubelet is
			// using CSI for all Migrated volume plugins. Then all the CSINode initialization
			// code can be dropped from Kubelet.
			// Kill the Kubelet process and allow it to restart to retry initialization
			klog.Fatalf("Failed to initialize CSINodeInfo after retrying")
		}
	}()
	return nil
}

// GetPluginName returns the unique name of this volume plugin
// implements: VolumePlugin
func (p *csiPlugin) GetPluginName() string {
	return CSIPluginName
}

// GetVolumeName returns a concatenated string of CSIVolumeSource.Driver<volNameSe>CSIVolumeSource.VolumeHandle
// That string value is used in Detach() to extract driver name and volumeName.
// implements: VolumePlugin
func (p *csiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	csi, err := getPVSourceFromSpec(spec)
	if err != nil {
		klog.Error(log("plugin.GetVolumeName failed to extract volume source from spec: %v", err))
		return "", err
	}

	// return driverName<separator>volumeHandle
	return fmt.Sprintf("%s%s%s", csi.Driver, volNameSep, csi.VolumeHandle), nil
}

// CanSupport states whether the plugin supports a given volume
// specification from the API.  The spec pointer should be considered
// const.
// implements: VolumePlugin
func (p *csiPlugin) CanSupport(spec *volume.Spec) bool {
	// TODO (vladimirvivien) CanSupport should also take into account
	// the availability/registration of specified Driver in the volume source
	if spec == nil {
		return false
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
		return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil) ||
			(spec.Volume != nil && spec.Volume.CSI != nil)
	}

	return spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil
}

// IsMigratedToCSI tests whether a CSIDriver implements this plugin's
// functionality
// implements: VolumePlugin
func (p *csiPlugin) IsMigratedToCSI() bool {
	return false
}

// RequiresRemount returns true if this plugin requires mount calls to be
// reexecuted. Atomically updating volumes, like Downward API, depend on
// this to update the contents of the volume.
// implements: VolumePlugin
func (p *csiPlugin) RequiresRemount() bool {
	return false
}

// NewMounter creates a new volume.Mounter from an API specification.
// Ownership of the spec pointer in *not* transferred.
// implements: VolumePlugin
func (p *csiPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {

	volSrc, pvSrc, err := getSourceFromSpec(spec)
	if err != nil {
		return nil, err
	}

	var (
		driverName   string
		volumeHandle string
		readOnly     bool
	)

	switch {
	case volSrc != nil && utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume):
		volumeHandle = makeVolumeHandle(string(pod.UID), spec.Name())
		driverName = volSrc.Driver
		if volSrc.ReadOnly != nil {
			readOnly = *volSrc.ReadOnly
		}
	case pvSrc != nil:
		driverName = pvSrc.Driver
		volumeHandle = pvSrc.VolumeHandle
		readOnly = spec.ReadOnly
	default:
		return nil, fmt.Errorf("volume source not found in volume.Spec")
	}

	driverMode, err := p.getDriverMode(spec)
	if err != nil {
		return nil, err
	}

	k8s := p.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("failed to get a kubernetes client"))
		return nil, errors.New("failed to get a Kubernetes client")
	}

	mounter := &csiMountMgr{
		plugin:       p,
		k8s:          k8s,
		spec:         spec,
		pod:          pod,
		podUID:       pod.UID,
		driverName:   csiDriverName(driverName),
		driverMode:   driverMode,
		volumeID:     volumeHandle,
		specVolumeID: spec.Name(),
		readOnly:     readOnly,
	}
	mounter.csiClientGetter.driverName = csiDriverName(driverName)
	mounter.csiClientGetter.clientCreator = p.newCsiDriverClient

	// Save volume info in pod dir
	dir := mounter.GetPath()
	dataDir := path.Dir(dir) // dropoff /mount at end

	if err := os.MkdirAll(dataDir, 0750); err != nil {
		klog.Error(log("failed to create dir %#v:  %v", dataDir, err))
		return nil, err
	}
	klog.V(4).Info(log("created path successfully [%s]", dataDir))

	// persist volume info data for teardown
	node := string(p.host.GetNodeName())
	volData := map[string]string{
		volDataKey.specVolID:  spec.Name(),
		volDataKey.volHandle:  volumeHandle,
		volDataKey.driverName: driverName,
		volDataKey.nodeName:   node,
		volDataKey.driverMode: string(driverMode),
	}

	attachID := getAttachmentName(volumeHandle, driverName, node)
	volData[volDataKey.attachmentID] = attachID

	if err := saveVolumeData(dataDir, volDataFileName, volData); err != nil {
		klog.Error(log("failed to save volume info data: %v", err))
		if err := os.RemoveAll(dataDir); err != nil {
			klog.Error(log("failed to remove dir after error [%s]: %v", dataDir, err))
			return nil, err
		}
		return nil, err
	}

	klog.V(4).Info(log("mounter created successfully"))

	return mounter, nil
}

// NewUnmounter creates a new volume.Unmounter from recoverable state.
// implements: VolumePlugin
func (p *csiPlugin) NewUnmounter(specName string, podUID types.UID) (volume.Unmounter, error) {
	klog.V(4).Infof(log("setting up unmounter for [name=%v, podUID=%v]", specName, podUID))

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
		klog.Error(log("unmounter failed to load volume data file [%s]: %v", dir, err))
		return nil, err
	}
	unmounter.driverName = csiDriverName(data[volDataKey.driverName])
	unmounter.volumeID = data[volDataKey.volHandle]
	unmounter.csiClientGetter.driverName = unmounter.driverName
	unmounter.csiClientGetter.clientCreator = p.newCsiDriverClient

	return unmounter, nil
}

// ConstructVolumeSpec constructs a volume spec based on the given volume name
// and mountPath. The spec may have incomplete information due to limited
// information from input. This function is used by volume manager to reconstruct
// volume spec by reading the volume directories from disk
// implements: VolumePlugin
func (p *csiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	klog.V(4).Info(log("plugin.ConstructVolumeSpec [pv.Name=%v, path=%v]", volumeName, mountPath))

	volData, err := loadVolumeData(mountPath, volDataFileName)
	if err != nil {
		klog.Error(log("plugin.ConstructVolumeSpec failed loading volume data using [%s]: %v", mountPath, err))
		return nil, err
	}

	klog.V(4).Info(log("plugin.ConstructVolumeSpec extracted [%#v]", volData))

	var spec *volume.Spec
	inlineEnabled := utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume)

	if inlineEnabled {
		mode := driverMode(volData[volDataKey.driverMode])
		switch {
		case mode == ephemeralDriverMode:
			spec = p.constructVolSourceSpec(volData[volDataKey.specVolID], volData[volDataKey.driverName])

		case mode == persistentDriverMode:
			fallthrough
		default:
			spec = p.constructPVSourceSpec(volData[volDataKey.specVolID], volData[volDataKey.driverName], volData[volDataKey.volHandle])
		}
	} else {
		spec = p.constructPVSourceSpec(volData[volDataKey.specVolID], volData[volDataKey.driverName], volData[volDataKey.volHandle])
	}

	return spec, nil
}

// constructVolSourceSpec constructs volume.Spec with CSIVolumeSource
func (p *csiPlugin) constructVolSourceSpec(volSpecName, driverName string) *volume.Spec {
	vol := &api.Volume{
		Name: volSpecName,
		VolumeSource: api.VolumeSource{
			CSI: &api.CSIVolumeSource{
				Driver: driverName,
			},
		},
	}
	return volume.NewSpecFromVolume(vol)
}

//constructPVSourceSpec constructs volume.Spec with CSIPersistentVolumeSource
func (p *csiPlugin) constructPVSourceSpec(volSpecName, driverName, volumeHandle string) *volume.Spec {
	fsMode := api.PersistentVolumeFilesystem
	pv := &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: volSpecName,
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volumeHandle,
				},
			},
			VolumeMode: &fsMode,
		},
	}
	return volume.NewSpecFromPersistentVolume(pv, false)
}

// SupportsMountOption returns true if volume plugins supports Mount options
// Specifying mount options in a volume plugin that doesn't support
// user specified mount options will result in error creating persistent volumes
// implements: VolumePlugin
func (p *csiPlugin) SupportsMountOption() bool {
	// TODO (vladimirvivien) use CSI VolumeCapability.MountVolume.mount_flags
	// to probe for the result for this method
	// (bswartz) Until the CSI spec supports probing, our only option is to
	// make plugins register their support for mount options or lack thereof
	// directly with kubernetes.
	return true
}

// SupportsBulkVolumeVerification checks if volume plugin type is capable
// of enabling bulk polling of all nodes. This can speed up verification of
// attached volumes by quite a bit, but underlying pluging must support it.
// implements: VolumePlugin
func (p *csiPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

// volume.AttachableVolumePlugin methods
var _ volume.AttachableVolumePlugin = &csiPlugin{}

var _ volume.DeviceMountableVolumePlugin = &csiPlugin{}

// NewAttacher initializes a new Attacher
// implements: AttachableVolumePlugin
func (p *csiPlugin) NewAttacher() (volume.Attacher, error) {
	k8s := p.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("unable to get kubernetes client from host"))
		return nil, errors.New("unable to get Kubernetes client")
	}

	return &csiAttacher{
		plugin:        p,
		k8s:           k8s,
		waitSleepTime: 1 * time.Second,
	}, nil
}

// NewDeviceMounter initializes a new DeviceMounter
// implements: DeviceMountablePlugin
func (p *csiPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return p.NewAttacher()
}

// NewDetacher initializes a new Detacher
// implements: AttachableVolumePlugin
func (p *csiPlugin) NewDetacher() (volume.Detacher, error) {
	k8s := p.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("unable to get kubernetes client from host"))
		return nil, errors.New("unable to get Kubernetes client")
	}

	return &csiAttacher{
		plugin:        p,
		k8s:           k8s,
		waitSleepTime: 1 * time.Second,
	}, nil
}

// CanAttach checks if the given volume spec is attachable
// implements: AttachableVolumePlugin
// TODO change CanAttach to return error to propagate ability
// to support Attachment or an error - see https://github.com/kubernetes/kubernetes/issues/74810
func (p *csiPlugin) CanAttach(spec *volume.Spec) bool {
	driverMode, err := p.getDriverMode(spec)
	if err != nil {
		return false
	}

	if driverMode == ephemeralDriverMode {
		klog.V(4).Info(log("driver ephemeral mode detected for spec %v", spec.Name))
		return false
	}

	pvSrc, err := getCSISourceFromSpec(spec)
	if err != nil {
		klog.Error(log("plugin.CanAttach failed to get info from spec: %s", err))
		return false
	}

	driverName := pvSrc.Driver

	skipAttach, err := p.skipAttach(driverName)
	if err != nil {
		klog.Error(log("plugin.CanAttach error when calling plugin.skipAttach for driver %s: %s", driverName, err))
		return false
	}

	return !skipAttach
}

// TODO (#75352) add proper logic to determine device moutability by inspecting the spec.
func (p *csiPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

// NewDeviceUnmounter initializes a new DeviceUnmounter
// implements: DeviceMountablePlugin
func (p *csiPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return p.NewDetacher()
}

// GetDeviceMountRefs finds all mount references to the path, returns a
// list of paths. Path could be a mountpoint path, device or a normal
// directory (for bind mount).
// implements: DeviceMountablePlugin
func (p *csiPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	m := p.host.GetMounter(p.GetPluginName())
	return m.GetMountRefs(deviceMountPath)
}

// BlockVolumePlugin methods
var _ volume.BlockVolumePlugin = &csiPlugin{}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
// Ownership of the spec pointer in *not* transferred.
// implements: BlocckVolumePlugin
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

	klog.V(4).Info(log("setting up block mapper for [volume=%v,driver=%v]", pvSource.VolumeHandle, pvSource.Driver))

	k8s := p.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("failed to get a kubernetes client"))
		return nil, errors.New("failed to get a Kubernetes client")
	}

	mapper := &csiBlockMapper{
		k8s:        k8s,
		plugin:     p,
		volumeID:   pvSource.VolumeHandle,
		driverName: csiDriverName(pvSource.Driver),
		readOnly:   readOnly,
		spec:       spec,
		specName:   spec.Name(),
		podUID:     podRef.UID,
	}
	mapper.csiClientGetter.driverName = csiDriverName(pvSource.Driver)
	mapper.csiClientGetter.clientCreator = p.newCsiDriverClient

	// Save volume info in pod dir
	dataDir := getVolumeDeviceDataDir(spec.Name(), p.host)

	if err := os.MkdirAll(dataDir, 0750); err != nil {
		klog.Error(log("failed to create data dir %s:  %v", dataDir, err))
		return nil, err
	}
	klog.V(4).Info(log("created path successfully [%s]", dataDir))

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
		klog.Error(log("failed to save volume info data: %v", err))
		if err := os.RemoveAll(dataDir); err != nil {
			klog.Error(log("failed to remove dir after error [%s]: %v", dataDir, err))
			return nil, err
		}
		return nil, err
	}

	return mapper, nil
}

// NewBlockVolumeUnmapper creates a new volume.BlockVolumeUnmapper from recoverable state.
// implements: BlockVolumePlugin
func (p *csiPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	if !p.blockEnabled {
		return nil, errors.New("CSIBlockVolume feature not enabled")
	}

	klog.V(4).Infof(log("setting up block unmapper for [Spec=%v, podUID=%v]", volName, podUID))
	unmapper := &csiBlockMapper{
		plugin:   p,
		podUID:   podUID,
		specName: volName,
	}

	// load volume info from file
	dataDir := getVolumeDeviceDataDir(unmapper.specName, p.host)
	data, err := loadVolumeData(dataDir, volDataFileName)
	if err != nil {
		klog.Error(log("unmapper failed to load volume data file [%s]: %v", dataDir, err))
		return nil, err
	}
	unmapper.driverName = csiDriverName(data[volDataKey.driverName])
	unmapper.volumeID = data[volDataKey.volHandle]
	unmapper.csiClientGetter.driverName = unmapper.driverName
	unmapper.csiClientGetter.clientCreator = p.newCsiDriverClient

	return unmapper, nil
}

// ConstructBlockVolumeSpec constructs a volume spec based on the given
// podUID, volume name and a pod device map path.
// The spec may have incomplete information due to limited information
// from input. This function is used by volume manager to reconstruct
// volume spec by reading the volume directories from disk.
// implements: BlockVolumePlugin
func (p *csiPlugin) ConstructBlockVolumeSpec(podUID types.UID, specVolName, mapPath string) (*volume.Spec, error) {
	if !p.blockEnabled {
		return nil, errors.New("CSIBlockVolume feature not enabled")
	}

	klog.V(4).Infof("plugin.ConstructBlockVolumeSpec [podUID=%s, specVolName=%s, path=%s]", string(podUID), specVolName, mapPath)

	dataDir := getVolumeDeviceDataDir(specVolName, p.host)
	volData, err := loadVolumeData(dataDir, volDataFileName)
	if err != nil {
		klog.Error(log("plugin.ConstructBlockVolumeSpec failed loading volume data using [%s]: %v", mapPath, err))
		return nil, err
	}

	klog.V(4).Info(log("plugin.ConstructBlockVolumeSpec extracted [%#v]", volData))

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

// skipAttach looks up CSIDriver object associated with driver name
// to determine if driver requies attachment volume operation
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

// getDriverMode returns the driver mode for the specified spec: {persistent|ephemeral}.
// 1) If mode cannot be determined, it will default to "persistent".
// 2) If Mode cannot be resolved to either {persistent | ephemeral}, an error is returned
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-storage/20190122-csi-inline-volumes.md
func (p *csiPlugin) getDriverMode(spec *volume.Spec) (driverMode, error) {
	// TODO (vladimirvivien) ultimately, mode will be retrieved from CSIDriver.Spec.Mode.
	// However, in alpha version, mode is determined by the volume source:
	// 1) if volume.Spec.Volume.CSI != nil -> mode is ephemeral
	// 2) if volume.Spec.PersistentVolume.Spec.CSI != nil -> persistent
	volSrc, _, err := getSourceFromSpec(spec)
	if err != nil {
		return "", err
	}

	if volSrc != nil && utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
		return ephemeralDriverMode, nil
	}
	return persistentDriverMode, nil
}

func (p *csiPlugin) getPublishContext(client clientset.Interface, handle, driver, nodeName string) (map[string]string, error) {
	skip, err := p.skipAttach(driver)
	if err != nil {
		return nil, err
	}
	if skip {
		return nil, nil
	}

	attachID := getAttachmentName(handle, driver, nodeName)

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	attachment, err := client.StorageV1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		return nil, err // This err already has enough context ("VolumeAttachment xyz not found")
	}

	if attachment == nil {
		err = errors.New("no existing VolumeAttachment found")
		return nil, err
	}
	return attachment.Status.AttachmentMetadata, nil
}

// for PluginHandler
func (p *csiPlugin) unregisterDriver(driverName string) error {
	p.drivers.Delete(driverName)

	if err := p.nim.UninstallCSIDriver(driverName); err != nil {
		klog.Errorf("Error uninstalling CSI driver: %v", err)
		return err
	}

	return nil
}

// Return the highest supported version
func highestSupportedVersion(versions []string) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, fmt.Errorf("CSI driver reporting empty array for supported versions")
	}

	// Sort by lowest to highest version
	sort.Slice(versions, func(i, j int) bool {
		parsedVersionI, err := utilversion.ParseGeneric(versions[i])
		if err != nil {
			// Push bad values to the bottom
			return true
		}

		parsedVersionJ, err := utilversion.ParseGeneric(versions[j])
		if err != nil {
			// Push bad values to the bottom
			return false
		}

		return parsedVersionI.LessThan(parsedVersionJ)
	})

	for i := len(versions) - 1; i >= 0; i-- {
		highestSupportedVersion, err := utilversion.ParseGeneric(versions[i])
		if err != nil {
			return nil, err
		}

		if highestSupportedVersion.Major() <= 1 {
			return highestSupportedVersion, nil
		}
	}

	return nil, fmt.Errorf("None of the CSI versions reported by this driver are supported")
}

// Only drivers that implement CSI 0.x are allowed to use deprecated socket dir.
func isDeprecatedSocketDirAllowed(versions []string) bool {
	for _, version := range versions {
		if isV0Version(version) {
			return true
		}
	}

	return false
}

func isV0Version(version string) bool {
	parsedVersion, err := utilversion.ParseGeneric(version)
	if err != nil {
		return false
	}

	return parsedVersion.Major() == 0
}

func isV1Version(version string) bool {
	parsedVersion, err := utilversion.ParseGeneric(version)
	if err != nil {
		return false
	}

	return parsedVersion.Major() == 1
}

// ToPluginHandler converts the generic volume plugin to a specific csiPlugin and
// returns that as a PluginHandler. In case the generic plugin is not a CSI
// plugin and thus not a PluginHandler, an error is returned.
func ToPluginHandler(genericPlugin volume.VolumePlugin) (pluginwatcher.PluginHandler, error) {
	csiPlugin, ok := genericPlugin.(*csiPlugin)
	if !ok {
		return nil, errors.New("volume plugin is not a CSI volume plugin")
	}

	return pluginwatcher.PluginHandler(csiPlugin), nil
}
