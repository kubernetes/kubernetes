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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/volume"
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
	}
}

func CreateTestClient() *fake.Clientset {
	fakeClient := &fake.Clientset{}

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
		return true, obj, nil
	})
	fakeClient.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.NodeList{}
		nodeNamePrefix := "mynode"
		namespace := "mynamespace"
		for i := 0; i < 5; i++ {
			nodeName := fmt.Sprintf("%s-%d", nodeNamePrefix, i)
			node := v1.Node{
				ObjectMeta: v1.ObjectMeta{
					Name:      nodeName,
					Namespace: namespace,
					Labels: map[string]string{
						"name": nodeName,
					},
				},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       TestPluginName + "/volumeName",
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

type testPlugins struct {
}

func (plugin *testPlugins) Init(host volume.VolumeHost) error {
	return nil
}

func (plugin *testPlugins) GetPluginName() string {
	return TestPluginName
}

func (plugin *testPlugins) GetVolumeName(spec *volume.Spec) (string, error) {
	return spec.Name(), nil
}

func (plugin *testPlugins) CanSupport(spec *volume.Spec) bool {
	return true
}

func (plugin *testPlugins) RequiresRemount() bool {
	return false
}

func (plugin *testPlugins) NewMounter(spec *volume.Spec, podRef *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, nil
}

func (plugin *testPlugins) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return nil, nil
}

func (plugin *testPlugins) ConstructVolumeSpecFromName(volumeName string) (*volume.Spec, error) {
	newVolume := &v1.Volume{
		Name:         volumeName,
		VolumeSource: v1.VolumeSource{},
	}
	return volume.NewSpecFromVolume(newVolume), nil
}

func (plugin *testPlugins) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	return nil, nil
}

func (plugin *testPlugins) NewAttacher() (volume.Attacher, error) {
	return nil, nil
}

func (plugin *testPlugins) NewDetacher() (volume.Detacher, error) {
	return nil, nil
}

func (plugin *testPlugins) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return []string{}, nil
}

func CreateTestPlugin() []volume.VolumePlugin {
	return []volume.VolumePlugin{&testPlugins{}}
}
