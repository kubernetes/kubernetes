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
	"fmt"
	"math/rand"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	fake_cloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/persistentvolume"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
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
	// Use higher QPS and Burst, there is a test for race condition below, which
	// creates many claims and default values were too low.
	binderClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000, Burst: 100000})
	recyclerClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000, Burst: 100000})
	testClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000, Burst: 100000})
	host := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)

	plugins := []volume.VolumePlugin{&volumetest.FakeVolumePlugin{"plugin-name", host, volume.VolumeConfig{}, volume.VolumeOptions{}, 0, 0, nil, nil, nil, nil}}
	cloud := &fake_cloud.FakeCloud{}

	binder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(binderClient, 10*time.Second)
	binder.Run()
	defer binder.Stop()

	recycler, _ := persistentvolumecontroller.NewPersistentVolumeRecycler(recyclerClient, 30*time.Second, 3, plugins, cloud)
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

	w, _ := testClient.PersistentVolumes().Watch(api.ListOptions{})
	defer w.Stop()

	_, _ = testClient.PersistentVolumes().Create(pv)
	_, _ = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)

	// wait until the binder pairs the volume and claim
	waitForPersistentVolumePhase(w, api.VolumeBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
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

	w, _ = testClient.PersistentVolumes().Watch(api.ListOptions{})
	defer w.Stop()

	_, _ = testClient.PersistentVolumes().Create(pv)
	_, _ = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)

	waitForPersistentVolumePhase(w, api.VolumeBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(w, api.VolumeReleased)

	for {
		event := <-w.ResultChan()
		if event.Type == watch.Deleted {
			break
		}
	}

	// test the race between claims and volumes.  ensure only a volume only binds to a single claim.
	deleteAllEtcdKeys()
	counter := 0
	maxClaims := 100
	claims := []*api.PersistentVolumeClaim{}
	for counter <= maxClaims {
		counter += 1
		clone, _ := conversion.NewCloner().DeepCopy(pvc)
		newPvc, _ := clone.(*api.PersistentVolumeClaim)
		newPvc.ObjectMeta = api.ObjectMeta{Name: fmt.Sprintf("fake-pvc-%d", counter)}
		claim, err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(newPvc)
		if err != nil {
			t.Fatal("Error creating newPvc: %v", err)
		}
		claims = append(claims, claim)
	}

	// putting a bind manually on a pv should only match the claim it is bound to
	rand.Seed(time.Now().Unix())
	claim := claims[rand.Intn(maxClaims-1)]
	claimRef, err := api.GetReference(claim)
	if err != nil {
		t.Fatalf("Unexpected error getting claimRef: %v", err)
	}
	pv.Spec.ClaimRef = claimRef

	pv, err = testClient.PersistentVolumes().Create(pv)
	if err != nil {
		t.Fatalf("Unexpected error creating pv: %v", err)
	}

	waitForPersistentVolumePhase(w, api.VolumeBound)

	pv, err = testClient.PersistentVolumes().Get(pv.Name)
	if err != nil {
		t.Fatalf("Unexpected error getting pv: %v", err)
	}
	if pv.Spec.ClaimRef == nil {
		t.Fatalf("Unexpected nil claimRef")
	}
	if pv.Spec.ClaimRef.Namespace != claimRef.Namespace || pv.Spec.ClaimRef.Name != claimRef.Name {
		t.Fatalf("Bind mismatch! Expected %s/%s but got %s/%s", claimRef.Namespace, claimRef.Name, pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name)
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
