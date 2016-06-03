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
	"net/http/httptest"
	"strconv"
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
	testClient, ctrl, watchPV, watchPVC := createClients(t, s)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	ctrl.Run()
	defer ctrl.Stop()

	// This PV will be claimed, released, and recycled.
	pv := createPV("fake-pv", "/tmp/foo", "10G", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}, api.PersistentVolumeReclaimRecycle)

	pvc := createPVC("fake-pvc", "5G", []api.PersistentVolumeAccessMode{api.ReadWriteOnce})

	_, err := testClient.PersistentVolumes().Create(pv)
	if err != nil {
		t.Errorf("Failed to create PersistentVolume: %v", err)
	}

	_, err = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)
	if err != nil {
		t.Errorf("Failed to create PersistentVolumeClaim: %v", err)
	}

	// wait until the controller pairs the volume and claim
	waitForPersistentVolumePhase(watchPV, api.VolumeBound)
	waitForPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(watchPV, api.VolumeReleased)
	waitForPersistentVolumePhase(watchPV, api.VolumeAvailable)

	// end of Recycler test.
	// Deleter test begins now.
	// tests are serial because running masters concurrently that delete keys may cause similar tests to time out

	deleteAllEtcdKeys()

	// change the reclamation policy of the PV for the next test
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimDelete

	_, err = testClient.PersistentVolumes().Create(pv)
	if err != nil {
		t.Errorf("Failed to create PersistentVolume: %v", err)
	}
	_, err = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)
	if err != nil {
		t.Errorf("Failed to create PersistentVolumeClaim: %v", err)
	}

	waitForPersistentVolumePhase(watchPV, api.VolumeBound)
	waitForPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(watchPV, api.VolumeReleased)

	for {
		event := <-watchPV.ResultChan()
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

	waitForPersistentVolumePhase(watchPV, api.VolumeBound)
	waitForPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)

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

func TestPersistentVolumeMultiPVs(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	testClient, controller, watchPV, watchPVC := createClients(t, s)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	controller.Run()
	defer controller.Stop()

	maxPVs := 100
	pvs := make([]*api.PersistentVolume, maxPVs)
	for i := 0; i < maxPVs; i++ {
		// This PV will be claimed, released, and deleted
		pvs[i] = createPV("pv-"+strconv.Itoa(i), "/tmp/foo"+strconv.Itoa(i), strconv.Itoa(i)+"G",
			[]api.PersistentVolumeAccessMode{api.ReadWriteOnce}, api.PersistentVolumeReclaimRetain)
	}

	pvc := createPVC("pvc-2", strconv.Itoa(maxPVs/2)+"G", []api.PersistentVolumeAccessMode{api.ReadWriteOnce})

	for i := 0; i < maxPVs; i++ {
		_, err := testClient.PersistentVolumes().Create(pvs[i])
		if err != nil {
			t.Errorf("Failed to create PersistentVolume %d: %v", i, err)
		}
	}
	t.Log("volumes created")

	_, err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)
	if err != nil {
		t.Errorf("Failed to create PersistentVolumeClaim: %v", err)
	}
	t.Log("claim created")

	// wait until the controller pairs the volume and claim
	waitForPersistentVolumePhase(watchPV, api.VolumeBound)
	t.Log("volume bound")
	waitForPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)
	t.Log("claim bound")

	// only one PV is bound
	bound := 0
	for i := 0; i < maxPVs; i++ {
		pv, err := testClient.PersistentVolumes().Get(pvs[i].Name)
		if err != nil {
			t.Fatalf("Unexpected error getting pv: %v", err)
		}
		if pv.Spec.ClaimRef == nil {
			continue
		}
		// found a bounded PV
		p := pv.Spec.Capacity[api.ResourceStorage]
		pvCap := p.Value()
		expectedCap := resource.MustParse(strconv.Itoa(maxPVs/2) + "G")
		expectedCapVal := expectedCap.Value()
		if pv.Spec.ClaimRef.Name != pvc.Name || pvCap != expectedCapVal {
			t.Fatalf("Bind mismatch! Expected %s capacity %d but got %s capacity %d", pvc.Name, expectedCapVal, pv.Spec.ClaimRef.Name, pvCap)
		}
		t.Logf("claim bounded to %s capacity %v", pv.Name, pv.Spec.Capacity[api.ResourceStorage])
		bound += 1
	}
	t.Log("volumes checked")

	if bound != 1 {
		t.Fatalf("Only 1 PV should be bound but got %d", bound)
	}

	// deleting a claim releases the volume
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}
	t.Log("claim deleted")

	waitForPersistentVolumePhase(watchPV, api.VolumeReleased)
	t.Log("volumes released")

	deleteAllEtcdKeys()
}

func TestPersistentVolumeMultiPVsDiffAccessModes(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	testClient, controller, watchPV, watchPVC := createClients(t, s)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	controller.Run()
	defer controller.Stop()

	// This PV will be claimed, released, and deleted
	pv_rwo := createPV("pv-rwo", "/tmp/foo", "10G",
		[]api.PersistentVolumeAccessMode{api.ReadWriteOnce}, api.PersistentVolumeReclaimRetain)
	pv_rwm := createPV("pv-rwm", "/tmp/bar", "10G",
		[]api.PersistentVolumeAccessMode{api.ReadWriteMany}, api.PersistentVolumeReclaimRetain)

	pvc := createPVC("pvc-rwm", "5G", []api.PersistentVolumeAccessMode{api.ReadWriteMany})

	_, err := testClient.PersistentVolumes().Create(pv_rwm)
	if err != nil {
		t.Errorf("Failed to create PersistentVolume: %v", err)
	}
	_, err = testClient.PersistentVolumes().Create(pv_rwo)
	if err != nil {
		t.Errorf("Failed to create PersistentVolume: %v", err)
	}
	t.Log("volumes created")

	_, err = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvc)
	if err != nil {
		t.Errorf("Failed to create PersistentVolumeClaim: %v", err)
	}
	t.Log("claim created")

	// wait until the controller pairs the volume and claim
	waitForPersistentVolumePhase(watchPV, api.VolumeBound)
	t.Log("volume bound")
	waitForPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)
	t.Log("claim bound")

	// only RWM PV is bound
	pv, err := testClient.PersistentVolumes().Get("pv-rwo")
	if err != nil {
		t.Fatalf("Unexpected error getting pv: %v", err)
	}
	if pv.Spec.ClaimRef != nil {
		t.Fatalf("ReadWriteOnce PV shouldn't be bound")
	}
	pv, err = testClient.PersistentVolumes().Get("pv-rwm")
	if err != nil {
		t.Fatalf("Unexpected error getting pv: %v", err)
	}
	if pv.Spec.ClaimRef == nil {
		t.Fatalf("ReadWriteMany PV should be bound")
	}
	if pv.Spec.ClaimRef.Name != pvc.Name {
		t.Fatalf("Bind mismatch! Expected %s but got %s", pvc.Name, pv.Spec.ClaimRef.Name)
	}

	// deleting a claim releases the volume
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}
	t.Log("claim deleted")

	waitForPersistentVolumePhase(watchPV, api.VolumeReleased)
	t.Log("volume released")

	deleteAllEtcdKeys()
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

func waitForPersistentVolumeClaimPhase(w watch.Interface, phase api.PersistentVolumeClaimPhase) {
	for {
		event := <-w.ResultChan()
		claim := event.Object.(*api.PersistentVolumeClaim)
		if claim.Status.Phase == phase {
			break
		}
	}
}

func createClients(t *testing.T, s *httptest.Server) (*clientset.Clientset, *persistentvolumecontroller.PersistentVolumeController, watch.Interface, watch.Interface) {
	// Use higher QPS and Burst, there is a test for race condition below, which
	// creates many claims and default values were too low.
	testClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000, Burst: 100000})
	host := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugins := []volume.VolumePlugin{&volumetest.FakeVolumePlugin{
		PluginName:             "plugin-name",
		Host:                   host,
		Config:                 volume.VolumeConfig{},
		LastProvisionerOptions: volume.VolumeOptions{},
		NewAttacherCallCount:   0,
		NewDetacherCallCount:   0,
		Mounters:               nil,
		Unmounters:             nil,
		Attachers:              nil,
		Detachers:              nil,
	}}
	cloud := &fake_cloud.FakeCloud{}
	ctrl := persistentvolumecontroller.NewPersistentVolumeController(testClient, 10*time.Second, nil, plugins, cloud, "", nil, nil, nil)

	watchPV, err := testClient.PersistentVolumes().Watch(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to watch PersistentVolumes: %v", err)
	}
	watchPVC, err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Watch(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to watch PersistentVolumeClaimss: %v", err)
	}

	return testClient, ctrl, watchPV, watchPVC
}

func createPV(name, path, cap string, mode []api.PersistentVolumeAccessMode, reclaim api.PersistentVolumeReclaimPolicy) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource:        api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: path}},
			Capacity:                      api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse(cap)},
			AccessModes:                   mode,
			PersistentVolumeReclaimPolicy: reclaim,
		},
	}
}

func createPVC(name, cap string, mode []api.PersistentVolumeAccessMode) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec: api.PersistentVolumeClaimSpec{
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse(cap)}},
			AccessModes: mode,
		},
	}
}
