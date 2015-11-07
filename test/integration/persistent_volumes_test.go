// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/persistentvolume"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/integration/framework"
)

func init() {
	requireEtcd()
}

func TestPersistentVolumeRecycler(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	binderClient := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Default.Version()})
	recyclerClient := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Default.Version()})
	testClient := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Default.Version()})

	binder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(binderClient, 1*time.Second)
	binder.Run()
	defer binder.Stop()

	recycler, _ := persistentvolumecontroller.NewPersistentVolumeRecycler(recyclerClient, 1*time.Second, []volume.VolumePlugin{&volume.FakeVolumePlugin{"plugin-name", volume.NewFakeVolumeHost("/tmp/fake", nil, nil)}})
	recycler.Run()
	defer recycler.Stop()

	// This PV will be claimed, released, and recycled.
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{Name: "fake-pv"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource:        api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/tmp/foo"}},
			Capacity:                      api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("10G")},
			AccessModes:                   []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
		},
	}

	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: "fake-pvc"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("5G")}},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
		},
	}

	w, _ := testClient.PersistentVolumes().Watch(labels.Everything(), fields.Everything(), api.ListOptions{})
	defer w.Stop()

	_, _ = testClient.PersistentVolumes().Create(pv)
	_, _ = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)

	// wait until the binder pairs the volume and claim
	waitForPersistentVolumePhase(w, api.VolumeBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(w, api.VolumeReleased)
	waitForPersistentVolumePhase(w, api.VolumeAvailable)

	// end of Recycler test.
	// Deleter test begins now.
	// tests are serial because running masters concurrently that delete keys may cause similar tests to time out

	deleteAllEtcdKeys()

	// change the reclamation policy of the PV for the next test
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimDelete

	w, _ = testClient.PersistentVolumes().Watch(labels.Everything(), fields.Everything(), api.ListOptions{})
	defer w.Stop()

	_, _ = testClient.PersistentVolumes().Create(pv)
	_, _ = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)

	waitForPersistentVolumePhase(w, api.VolumeBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(w, api.VolumeReleased)

	for {
		event := <-w.ResultChan()
		if event.Type == watch.Deleted {
			break
		}
	}
}

func waitForPersistentVolumePhase(w watch.Interface, phase api.PersistentVolumePhase) {
	for {
		event := <-w.ResultChan()
		volume := event.Object.(*api.PersistentVolume)
		if volume.Status.Phase == phase {
			break
		}
	}
}
