/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package persistentvolume

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/volume"
)

func TestFailedRecycling(t *testing.T) {
	pv := &api.PersistentVolume{
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("8Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/tmp/data02",
				},
			},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
			ClaimRef: &api.ObjectReference{
				Name:      "foo",
				Namespace: "bar",
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumeReleased,
		},
	}

	mockClient := &mockBinderClient{
		volume: pv,
	}

	// no Init called for pluginMgr and no plugins are available.  Volume should fail recycling.
	plugMgr := volume.VolumePluginMgr{}

	recycler := &PersistentVolumeRecycler{
		kubeClient: &testclient.Fake{},
		client:     mockClient,
		pluginMgr:  plugMgr,
	}

	err := recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Unexpected non-nil error: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumeFailed {
		t.Errorf("Expected %s but got %s", api.VolumeFailed, mockClient.volume.Status.Phase)
	}

	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimDelete
	err = recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Unexpected non-nil error: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumeFailed {
		t.Errorf("Expected %s but got %s", api.VolumeFailed, mockClient.volume.Status.Phase)
	}
}
