/*
Copyright 2016 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const TestPluginName = "kubernetes.io/testPlugin"

// GetTestVolumeSpec returns a test volume spec
func GetTestVolumeSpec(volumeName string, diskName v1.UniqueVolumeName) *volume.Spec {
	return &volume.Spec{
		Volume: &v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   string(diskName),
					FSType:   "fake",
					ReadOnly: false,
				},
			},
		},
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
			},
		},
	}
}

var extraPods *v1.PodList

func CreateTestClient() *fake.Clientset {
	fakeClient := &fake.Clientset{}

	extraPods = &v1.PodList{}
	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PodList{}
		podNamePrefix := "mypod"
		namespace := "mynamespace"
		for i := 0; i < 5; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := v1.Pod{
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					UID:       types.UID(podName),
					Namespace: namespace,
					Labels: map[string]string{
						"name": podName,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "containerName",
							Image: "containerImage",
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "volumeMountName",
									ReadOnly:  false,
									MountPath: "/mnt",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "volumeName",
							VolumeSource: v1.VolumeSource{
								GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
									PDName:   "pdName",
									FSType:   "ext4",
									ReadOnly: false,
								},
							},
						},
					},
					NodeName: "mynode",
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		for _, pod := range extraPods.Items {
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		createAction := action.(core.CreateAction)
		pod := createAction.GetObject().(*v1.Pod)
		extraPods.Items = append(extraPods.Items, *pod)
		return true, createAction.GetObject(), nil
	})
	fakeClient.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.NodeList{}
		nodeNamePrefix := "mynode"
		for i := 0; i < 5; i++ {
			var nodeName string
			if i != 0 {
				nodeName = fmt.Sprintf("%s-%d", nodeNamePrefix, i)
			} else {
				// We want also the "mynode" node since all the testing pods live there
				nodeName = nodeNamePrefix
			}
			node := v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
					Labels: map[string]string{
						"name": nodeName,
					},
					Annotations: map[string]string{
						volumehelper.ControllerManagedAttachAnnotation: "true",
					},
				},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       TestPluginName + "/lostVolumeName",
							DevicePath: "fake/path",
						},
					},
				},
				Spec: v1.NodeSpec{ExternalID: string(nodeName)},
			}
			obj.Items = append(obj.Items, node)
		}
		return true, obj, nil
	})

	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))

	return fakeClient
}

// NewPod returns a test pod object
func NewPod(uid, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(uid),
			Name:      name,
			Namespace: name,
		},
	}
}

// NewPod returns a test pod object
func NewPodWithVolume(podName, volumeName, nodeName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(podName),
			Name:      podName,
			Namespace: "mynamespace",
			Labels: map[string]string{
				"name": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "containerName",
					Image: "containerImage",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "volumeMountName",
							ReadOnly:  false,
							MountPath: "/mnt",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName:   "pdName",
							FSType:   "ext4",
							ReadOnly: false,
						},
					},
				},
			},
			NodeName: nodeName,
		},
	}
}

type TestPlugin struct {
	ErrorEncountered  bool
	attachedVolumeMap map[string][]string
	detachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (plugin *TestPlugin) Init(host volume.VolumeHost) error {
	return nil
}

func (plugin *TestPlugin) GetPluginName() string {
	return TestPluginName
}

func (plugin *TestPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		glog.Errorf("GetVolumeName called with nil volume spec")
		plugin.ErrorEncountered = true
	}
	return spec.Name(), nil
}

func (plugin *TestPlugin) CanSupport(spec *volume.Spec) bool {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		glog.Errorf("CanSupport called with nil volume spec")
		plugin.ErrorEncountered = true
	}
	return true
}

func (plugin *TestPlugin) RequiresRemount() bool {
	return false
}

func (plugin *TestPlugin) NewMounter(spec *volume.Spec, podRef *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		glog.Errorf("NewMounter called with nil volume spec")
		plugin.ErrorEncountered = true
	}
	return nil, nil
}

func (plugin *TestPlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return nil, nil
}

func (plugin *TestPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	fakeVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName:   "pdName",
				FSType:   "ext4",
				ReadOnly: false,
			},
		},
	}
	return volume.NewSpecFromVolume(fakeVolume), nil
}

func (plugin *TestPlugin) NewAttacher() (volume.Attacher, error) {
	attacher := testPluginAttacher{
		ErrorEncountered:  &plugin.ErrorEncountered,
		attachedVolumeMap: plugin.attachedVolumeMap,
		pluginLock:        plugin.pluginLock,
	}
	return &attacher, nil
}

func (plugin *TestPlugin) NewDetacher() (volume.Detacher, error) {
	detacher := testPluginDetacher{
		detachedVolumeMap: plugin.detachedVolumeMap,
		pluginLock:        plugin.pluginLock,
	}
	return &detacher, nil
}

func (plugin *TestPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return []string{}, nil
}

func (plugin *TestPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *TestPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *TestPlugin) GetErrorEncountered() bool {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	return plugin.ErrorEncountered
}

func (plugin *TestPlugin) GetAttachedVolumes() map[string][]string {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	ret := make(map[string][]string)
	for nodeName, volumeList := range plugin.attachedVolumeMap {
		ret[nodeName] = make([]string, len(volumeList))
		copy(ret[nodeName], volumeList)
	}
	return ret
}

func (plugin *TestPlugin) GetDetachedVolumes() map[string][]string {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	ret := make(map[string][]string)
	for nodeName, volumeList := range plugin.detachedVolumeMap {
		ret[nodeName] = make([]string, len(volumeList))
		copy(ret[nodeName], volumeList)
	}
	return ret
}

func CreateTestPlugin() []volume.VolumePlugin {
	attachedVolumes := make(map[string][]string)
	detachedVolumes := make(map[string][]string)
	return []volume.VolumePlugin{&TestPlugin{
		ErrorEncountered:  false,
		attachedVolumeMap: attachedVolumes,
		detachedVolumeMap: detachedVolumes,
		pluginLock:        &sync.RWMutex{},
	}}
}

// Attacher
type testPluginAttacher struct {
	ErrorEncountered  *bool
	attachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (attacher *testPluginAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		glog.Errorf("Attach called with nil volume spec")
		return "", fmt.Errorf("Attach called with nil volume spec")
	}
	attacher.attachedVolumeMap[string(nodeName)] = append(attacher.attachedVolumeMap[string(nodeName)], spec.Name())
	return spec.Name(), nil
}

func (attacher *testPluginAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	return nil, nil
}

func (attacher *testPluginAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		glog.Errorf("WaitForAttach called with nil volume spec")
		return "", fmt.Errorf("WaitForAttach called with nil volume spec")
	}
	fakePath := fmt.Sprintf("%s/%s", devicePath, spec.Name())
	return fakePath, nil
}

func (attacher *testPluginAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		glog.Errorf("GetDeviceMountPath called with nil volume spec")
		return "", fmt.Errorf("GetDeviceMountPath called with nil volume spec")
	}
	return "", nil
}

func (attacher *testPluginAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		glog.Errorf("MountDevice called with nil volume spec")
		return fmt.Errorf("MountDevice called with nil volume spec")
	}
	return nil
}

// Detacher
type testPluginDetacher struct {
	detachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (detacher *testPluginDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	detacher.pluginLock.Lock()
	defer detacher.pluginLock.Unlock()
	detacher.detachedVolumeMap[string(nodeName)] = append(detacher.detachedVolumeMap[string(nodeName)], volumeName)
	return nil
}

func (detacher *testPluginDetacher) UnmountDevice(deviceMountPath string) error {
	return nil
}
