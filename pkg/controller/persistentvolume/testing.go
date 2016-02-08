/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
)

func makeTestVolume() *api.PersistentVolume {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo",
				},
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumePending,
			Conditions: []api.PersistentVolumeCondition{
				{
					Type:   api.PersistentVolumeBound,
					Status: api.ConditionFalse,
				},
			},
		},
	}
	pv.ObjectMeta.SelfLink = testapi.Default.SelfLink("pv", "")
	return pv
}
func makeTestClaim() *api.PersistentVolumeClaim {
	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "c1",
			Namespace: "ns",
			UID:       "123abc",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
			Conditions: []api.PersistentVolumeClaimCondition{
				{
					Type:   api.PersistentVolumeClaimBound,
					Status: api.ConditionFalse,
				},
			},
		},
	}
	claim.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")
	return claim
}

func makeTestTaskHost() *taskHost {
	taskHost, _ := newTaskHost(
		&mockTaskClient{},
		NewPersistentVolumeOrderedIndex(),
		cache.NewStore(controller.KeyFunc),
		workqueue.New(),
		volume.ProbeVolumePlugins(volume.VolumeConfig{}),
		&fakecloud.FakeCloud{})
	return taskHost
}

func makeTestTaskContext() *taskContext {
	return newTaskContext(makeTestTaskHost())
}

var _ taskClient = &mockTaskClient{}

type mockTaskClient struct {
	volume *api.PersistentVolume
	claim  *api.PersistentVolumeClaim
}

func (c *mockTaskClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.volume, nil
}

func (c *mockTaskClient) CreatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	if pv.GenerateName != "" && pv.Name == "" {
		pv.Name = fmt.Sprintf(pv.GenerateName, util.NewUUID())
	}
	c.volume = pv
	return c.volume, nil
}

func (c *mockTaskClient) ListPersistentVolumes(options api.ListOptions) (*api.PersistentVolumeList, error) {
	return &api.PersistentVolumeList{
		Items: []api.PersistentVolume{*c.volume},
	}, nil
}

func (c *mockTaskClient) WatchPersistentVolumes(options api.ListOptions) (watch.Interface, error) {
	return watch.NewFake(), nil
}

func (c *mockTaskClient) UpdatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.CreatePersistentVolume(pv)
}

func (c *mockTaskClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	c.volume = nil
	return nil
}

func (c *mockTaskClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	c.volume = volume
	return c.volume, nil
}

func (c *mockTaskClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	if c.claim != nil {
		return c.claim, nil
	} else {
		return nil, errors.NewNotFound(api.Resource("persistentvolumes"), name)
	}
}

func (c *mockTaskClient) ListPersistentVolumeClaims(namespace string, options api.ListOptions) (*api.PersistentVolumeClaimList, error) {
	return &api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{*c.claim},
	}, nil
}

func (c *mockTaskClient) WatchPersistentVolumeClaims(namespace string, options api.ListOptions) (watch.Interface, error) {
	return watch.NewFake(), nil
}

func (c *mockTaskClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.claim = claim
	return c.claim, nil
}

func (c *mockTaskClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.claim = claim
	return claim, nil
}

func (c *mockTaskClient) GetKubeClient() clientset.Interface {
	return nil
}
