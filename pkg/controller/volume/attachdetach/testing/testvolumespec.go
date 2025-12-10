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

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

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

func CreateTestClient() *fake.Clientset {
	var extraPods *v1.PodList
	var volumeAttachments *storagev1.VolumeAttachmentList
	var pvs *v1.PersistentVolumeList
	var nodes *v1.NodeList

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
									ReadOnly:  true,
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
									PDName: "pdName",
									FSType: "ext4",
									// Make the translated volume allow Multi-Attach.
									ReadOnly: true,
								},
							},
						},
					},
					NodeName: "mynode",
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		obj.Items = append(obj.Items, extraPods.Items...)
		return true, obj, nil
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		createAction := action.(core.CreateAction)
		pod := createAction.GetObject().(*v1.Pod)
		extraPods.Items = append(extraPods.Items, *pod)
		return true, createAction.GetObject(), nil
	})
	nodes = &v1.NodeList{}
	nodeNamePrefix := "mynode"
	for i := 0; i < 5; i++ {
		var nodeName string
		if i != 0 {
			nodeName = fmt.Sprintf("%s-%d", nodeNamePrefix, i)
		} else {
			// We want also the "mynode" node since all the testing pods live there
			nodeName = nodeNamePrefix
		}
		attachVolumeToNode(nodes, "lostVolumeName", nodeName, false)
	}
	attachVolumeToNode(nodes, "inUseVolume", nodeNamePrefix, true)
	fakeClient.AddReactor("update", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		updateAction := action.(core.UpdateAction)
		node := updateAction.GetObject().(*v1.Node)
		for index, n := range nodes.Items {
			if n.Name == node.Name {
				nodes.Items[index] = *node
			}
		}
		return true, updateAction.GetObject(), nil
	})
	fakeClient.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.NodeList{}
		obj.Items = append(obj.Items, nodes.Items...)
		return true, obj, nil
	})
	fakeClient.AddReactor("list", "csinodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &storagev1.CSINodeList{}
		for _, node := range nodes.Items {
			csiNode := storagev1.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						// All the in-tree plugins have been migrated to CSI since v1.27.
						// So hardcoding the migrated plugins here.
						"storage.alpha.kubernetes.io/migrated-plugins": "kubernetes.io/aws-ebs,kubernetes.io/azure-disk,kubernetes.io/azure-file,kubernetes.io/cinder,kubernetes.io/gce-pd,kubernetes.io/vsphere-volume",
					},
					Name: node.Name,
				},
			}
			obj.Items = append(obj.Items, csiNode)
		}
		return true, obj, nil
	})
	volumeAttachments = &storagev1.VolumeAttachmentList{}
	fakeClient.AddReactor("list", "volumeattachments", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &storagev1.VolumeAttachmentList{}
		obj.Items = append(obj.Items, volumeAttachments.Items...)
		return true, obj, nil
	})
	fakeClient.AddReactor("create", "volumeattachments", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		createAction := action.(core.CreateAction)
		va := createAction.GetObject().(*storagev1.VolumeAttachment)
		volumeAttachments.Items = append(volumeAttachments.Items, *va)
		return true, createAction.GetObject(), nil
	})

	pvs = &v1.PersistentVolumeList{}
	fakeClient.AddReactor("list", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PersistentVolumeList{}
		obj.Items = append(obj.Items, pvs.Items...)
		return true, obj, nil
	})
	fakeClient.AddReactor("create", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		createAction := action.(core.CreateAction)
		pv := createAction.GetObject().(*v1.PersistentVolume)
		pvs.Items = append(pvs.Items, *pv)
		return true, createAction.GetObject(), nil
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

// NewPodWithVolume returns a test pod object
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
							ReadOnly:  true,
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
							PDName: "pdName",
							FSType: "ext4",
							// Make the translated volume allow Multi-Attach.
							ReadOnly: true,
						},
					},
				},
			},
			NodeName: nodeName,
		},
	}
}

// Returns a volumeAttachment object
func NewVolumeAttachment(vaName, pvName, nodeName string, status bool) *storagev1.VolumeAttachment {
	return &storagev1.VolumeAttachment{

		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(vaName),
			Name: vaName,
		},
		Spec: storagev1.VolumeAttachmentSpec{
			Attacher: "test.storage.gke.io",
			NodeName: nodeName,
			Source: storagev1.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
		},
		Status: storagev1.VolumeAttachmentStatus{
			Attached: status,
		},
	}
}

// Returns a persistentVolume object
func NewPV(pvName, volumeName string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(pvName),
			Name: pvName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: volumeName,
				},
			},
		},
	}
}

// Returns an NFS PV. This can be used for an in-tree volume that is not migrated (unlike NewPV, which uses the GCE persistent disk).
func NewNFSPV(pvName, volumeName string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(pvName),
			Name: pvName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				NFS: &v1.NFSVolumeSource{
					Server: volumeName,
				},
			},
		},
	}
}

func attachVolumeToNode(nodes *v1.NodeList, volumeName, nodeName string, inUse bool) {
	// if nodeName exists, get the object.. if not create node object
	var node *v1.Node
	for i := range nodes.Items {
		curNode := &nodes.Items[i]
		if curNode.ObjectMeta.Name == nodeName {
			node = curNode
			break
		}
	}
	if node == nil {
		nodes.Items = append(nodes.Items, v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
				Labels: map[string]string{
					"name": nodeName,
				},
				Annotations: map[string]string{
					util.ControllerManagedAttachAnnotation: "true",
				},
			},
		})
		node = &nodes.Items[len(nodes.Items)-1]
	}
	uniqueVolumeName := v1.UniqueVolumeName(TestPluginName + "/" + volumeName)
	volumeAttached := v1.AttachedVolume{
		Name:       uniqueVolumeName,
		DevicePath: "fake/path",
	}
	node.Status.VolumesAttached = append(node.Status.VolumesAttached, volumeAttached)

	if inUse {
		node.Status.VolumesInUse = append(node.Status.VolumesInUse, uniqueVolumeName)
	}
}
