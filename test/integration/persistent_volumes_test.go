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
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/golang/glog"

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

// Several tests in this file are configurable by enviroment variables:
// KUBE_INTEGRATION_PV_OBJECTS - nr. of PVs/PVCs to be created
//      (100 by default)
// KUBE_INTEGRATION_PV_SYNC_PERIOD - volume controller sync period
//      (10s by default)
// KUBE_INTEGRATION_PV_END_SLEEP - for how long should
//      TestPersistentVolumeMultiPVsPVCs sleep when it's finished (0s by
//      default). This is useful to test how long does it take for periodic sync
//      to process bound PVs/PVCs.
//
const defaultObjectCount = 100
const defaultSyncPeriod = 10 * time.Second

func getObjectCount() int {
	objectCount := defaultObjectCount
	if s := os.Getenv("KUBE_INTEGRATION_PV_OBJECTS"); s != "" {
		var err error
		objectCount, err = strconv.Atoi(s)
		if err != nil {
			glog.Fatalf("cannot parse value of KUBE_INTEGRATION_PV_OBJECTS: %v", err)
		}
	}
	glog.V(2).Infof("using KUBE_INTEGRATION_PV_OBJECTS=%d", objectCount)
	return objectCount
}

func getSyncPeriod() time.Duration {
	period := defaultSyncPeriod
	if s := os.Getenv("KUBE_INTEGRATION_PV_SYNC_PERIOD"); s != "" {
		var err error
		period, err = time.ParseDuration(s)
		if err != nil {
			glog.Fatalf("cannot parse value of KUBE_INTEGRATION_PV_SYNC_PERIOD: %v", err)
		}
	}
	glog.V(2).Infof("using KUBE_INTEGRATION_PV_SYNC_PERIOD=%v", period)
	return period
}

func testSleep() {
	var period time.Duration
	if s := os.Getenv("KUBE_INTEGRATION_PV_END_SLEEP"); s != "" {
		var err error
		period, err = time.ParseDuration(s)
		if err != nil {
			glog.Fatalf("cannot parse value of KUBE_INTEGRATION_PV_END_SLEEP: %v", err)
		}
	}
	glog.V(2).Infof("using KUBE_INTEGRATION_PV_END_SLEEP=%v", period)
	if period != 0 {
		time.Sleep(period)
		glog.V(2).Infof("sleep finished")
	}
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
	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeBound)
	waitForPersistentVolumeClaimPhase(testClient, pvc.Name, watchPVC, api.ClaimBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeReleased)
	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeAvailable)

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
	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeBound)
	waitForPersistentVolumeClaimPhase(testClient, pvc.Name, watchPVC, api.ClaimBound)

	// deleting a claim releases the volume, after which it can be recycled
	if err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Delete(pvc.Name, nil); err != nil {
		t.Errorf("error deleting claim %s", pvc.Name)
	}

	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeReleased)

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

	waitForPersistentVolumePhase(testClient, pv.Name, watchPV, api.VolumeBound)
	waitForAnyPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)

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

// TestPersistentVolumeMultiPVs tests binding of one PVC to 100 PVs with
// different size.
func TestPersistentVolumeMultiPVs(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	testClient, controller, watchPV, watchPVC := createClients(t, s)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	controller.Run()
	defer controller.Stop()

	maxPVs := getObjectCount()
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

	// wait until the binder pairs the claim with a volume
	waitForAnyPersistentVolumePhase(watchPV, api.VolumeBound)
	t.Log("volume bound")
	waitForPersistentVolumeClaimPhase(testClient, pvc.Name, watchPVC, api.ClaimBound)
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

	waitForAnyPersistentVolumePhase(watchPV, api.VolumeReleased)
	t.Log("volumes released")

	deleteAllEtcdKeys()
}

// TestPersistentVolumeMultiPVsPVCs tests binding of 100 PVC to 100 PVs.
// This test is configurable by KUBE_INTEGRATION_PV_* variables.
func TestPersistentVolumeMultiPVsPVCs(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	testClient, binder, watchPV, watchPVC := createClients(t, s)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	binder.Run()
	defer binder.Stop()

	objCount := getObjectCount()
	pvs := make([]*api.PersistentVolume, objCount)
	pvcs := make([]*api.PersistentVolumeClaim, objCount)
	for i := 0; i < objCount; i++ {
		// This PV will be claimed, released, and deleted
		pvs[i] = createPV("pv-"+strconv.Itoa(i), "/tmp/foo"+strconv.Itoa(i), "1G",
			[]api.PersistentVolumeAccessMode{api.ReadWriteOnce}, api.PersistentVolumeReclaimRetain)
		pvcs[i] = createPVC("pvc-"+strconv.Itoa(i), "1G", []api.PersistentVolumeAccessMode{api.ReadWriteOnce})
	}

	// Create PVs first
	glog.V(2).Infof("TestPersistentVolumeMultiPVsPVCs: start")

	// Create the volumes in a separate goroutine to pop events from
	// watchPV early - it seems it has limited capacity and it gets stuck
	// with >3000 volumes.
	go func() {
		for i := 0; i < objCount; i++ {
			_, _ = testClient.PersistentVolumes().Create(pvs[i])
		}
	}()
	// Wait for them to get Available
	for i := 0; i < objCount; i++ {
		waitForAnyPersistentVolumePhase(watchPV, api.VolumeAvailable)
		glog.V(1).Infof("%d volumes available", i+1)
	}
	glog.V(2).Infof("TestPersistentVolumeMultiPVsPVCs: volumes are Available")

	// Create the claims, again in a separate goroutine.
	go func() {
		for i := 0; i < objCount; i++ {
			_, _ = testClient.PersistentVolumeClaims(api.NamespaceDefault).Create(pvcs[i])
		}
	}()

	// wait until the binder pairs all volumes
	for i := 0; i < objCount; i++ {
		waitForAnyPersistentVolumeClaimPhase(watchPVC, api.ClaimBound)
		glog.V(1).Infof("%d claims bound", i+1)
	}
	glog.V(2).Infof("TestPersistentVolumeMultiPVsPVCs: claims are bound")

	// check that everything is bound to something
	for i := 0; i < objCount; i++ {
		pv, err := testClient.PersistentVolumes().Get(pvs[i].Name)
		if err != nil {
			t.Fatalf("Unexpected error getting pv: %v", err)
		}
		if pv.Spec.ClaimRef == nil {
			t.Fatalf("PV %q is not bound", pv.Name)
		}
		glog.V(2).Infof("PV %q is bound to PVC %q", pv.Name, pv.Spec.ClaimRef.Name)

		pvc, err := testClient.PersistentVolumeClaims(api.NamespaceDefault).Get(pvcs[i].Name)
		if err != nil {
			t.Fatalf("Unexpected error getting pvc: %v", err)
		}
		if pvc.Spec.VolumeName == "" {
			t.Fatalf("PVC %q is not bound", pvc.Name)
		}
		glog.V(2).Infof("PVC %q is bound to PV %q", pvc.Name, pvc.Spec.VolumeName)
	}
	testSleep()
	deleteAllEtcdKeys()
}

// TestPersistentVolumeMultiPVsDiffAccessModes tests binding of one PVC to two
// PVs with different access modes.
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
	waitForAnyPersistentVolumePhase(watchPV, api.VolumeBound)
	t.Log("volume bound")
	waitForPersistentVolumeClaimPhase(testClient, pvc.Name, watchPVC, api.ClaimBound)
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

	waitForAnyPersistentVolumePhase(watchPV, api.VolumeReleased)
	t.Log("volume released")

	deleteAllEtcdKeys()
}

func waitForPersistentVolumePhase(client *clientset.Clientset, pvName string, w watch.Interface, phase api.PersistentVolumePhase) {
	// Check if the volume is already in requested phase
	volume, err := client.Core().PersistentVolumes().Get(pvName)
	if err == nil && volume.Status.Phase == phase {
		return
	}

	// Wait for the phase
	for {
		event := <-w.ResultChan()
		volume, ok := event.Object.(*api.PersistentVolume)
		if !ok {
			continue
		}
		if volume.Status.Phase == phase && volume.Name == pvName {
			glog.V(2).Infof("volume %q is %s", volume.Name, phase)
			break
		}
	}
}

func waitForPersistentVolumeClaimPhase(client *clientset.Clientset, claimName string, w watch.Interface, phase api.PersistentVolumeClaimPhase) {
	// Check if the claim is already in requested phase
	claim, err := client.Core().PersistentVolumeClaims(api.NamespaceDefault).Get(claimName)
	if err == nil && claim.Status.Phase == phase {
		return
	}

	// Wait for the phase
	for {
		event := <-w.ResultChan()
		claim, ok := event.Object.(*api.PersistentVolumeClaim)
		if !ok {
			continue
		}
		if claim.Status.Phase == phase && claim.Name == claimName {
			glog.V(2).Infof("claim %q is %s", claim.Name, phase)
			break
		}
	}
}

func waitForAnyPersistentVolumePhase(w watch.Interface, phase api.PersistentVolumePhase) {
	for {
		event := <-w.ResultChan()
		volume, ok := event.Object.(*api.PersistentVolume)
		if !ok {
			continue
		}
		if volume.Status.Phase == phase {
			glog.V(2).Infof("volume %q is %s", volume.Name, phase)
			break
		}
	}
}

func waitForAnyPersistentVolumeClaimPhase(w watch.Interface, phase api.PersistentVolumeClaimPhase) {
	for {
		event := <-w.ResultChan()
		claim, ok := event.Object.(*api.PersistentVolumeClaim)
		if !ok {
			continue
		}
		if claim.Status.Phase == phase {
			glog.V(2).Infof("claim %q is %s", claim.Name, phase)
			break
		}
	}
}

func createClients(t *testing.T, s *httptest.Server) (*clientset.Clientset, *persistentvolumecontroller.PersistentVolumeController, watch.Interface, watch.Interface) {
	// Use higher QPS and Burst, there is a test for race conditions which
	// creates many objects and default values were too low.
	binderClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000000, Burst: 1000000})
	testClient := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: 1000000, Burst: 1000000})

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

	syncPeriod := getSyncPeriod()
	ctrl := persistentvolumecontroller.NewPersistentVolumeController(binderClient, syncPeriod, nil, plugins, cloud, "", nil, nil, nil)

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
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PersistentVolumeClaimSpec{
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse(cap)}},
			AccessModes: mode,
		},
	}
}
