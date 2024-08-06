// /*
// Copyright 2024 The Kubernetes Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */

package demo

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"

	"k8s.io/klog/v2"
)

const defaultObjectCount = 2
const defaultSyncPeriod = 1 * time.Second

const provisionerPluginName = "kubernetes.io/mock-provisioner"

func getObjectCount() int {
	objectCount := defaultObjectCount
	if s := os.Getenv("KUBE_INTEGRATION_PV_OBJECTS"); s != "" {
		var err error
		objectCount, err = strconv.Atoi(s)
		if err != nil {
			klog.Fatalf("cannot parse value of KUBE_INTEGRATION_PV_OBJECTS: %v", err)
		}
	}
	klog.V(2).Infof("using KUBE_INTEGRATION_PV_OBJECTS=%d", objectCount)
	return objectCount
}

func getSyncPeriod(syncPeriod time.Duration) time.Duration {
	period := syncPeriod
	if s := os.Getenv("KUBE_INTEGRATION_PV_SYNC_PERIOD"); s != "" {
		var err error
		period, err = time.ParseDuration(s)
		if err != nil {
			klog.Fatalf("cannot parse value of KUBE_INTEGRATION_PV_SYNC_PERIOD: %v", err)
		}
	}
	klog.V(2).Infof("using KUBE_INTEGRATION_PV_SYNC_PERIOD=%v", period)
	return period
}

func waitForPersistentVolumeClaimPhase(client *clientset.Clientset, claimName, namespace string, w watch.Interface, phase v1.PersistentVolumeClaimPhase) {
	// Check if the claim is already in requested phase
	claim, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(context.TODO(), claimName, metav1.GetOptions{})
	if err == nil && claim.Status.Phase == phase {
		return
	}

	// Wait for the phase
	for {
		event := <-w.ResultChan()
		claim, ok := event.Object.(*v1.PersistentVolumeClaim)
		if !ok {
			continue
		}
		if claim.Status.Phase == phase && claim.Name == claimName {
			klog.V(2).Infof("claim %q is %s", claim.Name, phase)
			break
		}
	}
}

func waitForAnyPersistentVolumePhase(w watch.Interface, phase v1.PersistentVolumePhase) {
	for {
		event := <-w.ResultChan()
		volume, ok := event.Object.(*v1.PersistentVolume)
		if !ok {
			continue
		}
		if volume.Status.Phase == phase {
			klog.V(2).Infof("volume %q is %s", volume.Name, phase)
			break
		}
	}
}

func createClients(ctx context.Context, namespaceName string, t *testing.T, s *kubeapiservertesting.TestServer, syncPeriod time.Duration) (*clientset.Clientset, *persistentvolumecontroller.PersistentVolumeController, informers.SharedInformerFactory, watch.Interface, watch.Interface) {
	// Use higher QPS and Burst, there is a test for race conditions which
	// creates many objects and default values were too low.
	binderConfig := restclient.CopyConfig(s.ClientConfig)
	binderConfig.QPS = 1000000
	binderConfig.Burst = 1000000
	binderClient := clientset.NewForConfigOrDie(binderConfig)
	testConfig := restclient.CopyConfig(s.ClientConfig)
	testConfig.QPS = 1000000
	testConfig.Burst = 1000000
	testClient := clientset.NewForConfigOrDie(testConfig)

	host := volumetest.NewFakeVolumeHost(t, "/tmp/fake", nil, nil)
	plugin := &volumetest.FakeVolumePlugin{
		PluginName:             provisionerPluginName,
		Host:                   host,
		Config:                 volume.VolumeConfig{},
		LastProvisionerOptions: volume.VolumeOptions{},
		NewAttacherCallCount:   0,
		NewDetacherCallCount:   0,
		Mounters:               nil,
		Unmounters:             nil,
		Attachers:              nil,
		Detachers:              nil,
	}
	plugins := []volume.VolumePlugin{plugin}
	informers := informers.NewSharedInformerFactory(testClient, getSyncPeriod(syncPeriod))
	ctrl, err := persistentvolumecontroller.NewController(
		ctx,
		persistentvolumecontroller.ControllerParameters{
			KubeClient:                binderClient,
			SyncPeriod:                getSyncPeriod(syncPeriod),
			VolumePlugins:             plugins,
			VolumeInformer:            informers.Core().V1().PersistentVolumes(),
			ClaimInformer:             informers.Core().V1().PersistentVolumeClaims(),
			ClassInformer:             informers.Storage().V1().StorageClasses(),
			PodInformer:               informers.Core().V1().Pods(),
			NodeInformer:              informers.Core().V1().Nodes(),
			EnableDynamicProvisioning: true,
		})
	if err != nil {
		t.Fatalf("Failed to construct PersistentVolumes: %v", err)
	}

	watchPV, err := testClient.CoreV1().PersistentVolumes().Watch(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to watch PersistentVolumes: %v", err)
	}
	watchPVC, err := testClient.CoreV1().PersistentVolumeClaims(namespaceName).Watch(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to watch PersistentVolumeClaims: %v", err)
	}

	return testClient, ctrl, informers, watchPV, watchPVC
}

func createPV(name, path, cap string, mode []v1.PersistentVolumeAccessMode, reclaim v1.PersistentVolumeReclaimPolicy) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource:        v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: path}},
			Capacity:                      v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse(cap)},
			AccessModes:                   mode,
			PersistentVolumeReclaimPolicy: reclaim,
		},
	}
}

func createPVWithStorageClass(name, path, cap, scName string, mode []v1.PersistentVolumeAccessMode, reclaim v1.PersistentVolumeReclaimPolicy) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource:        v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: path}},
			Capacity:                      v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse(cap)},
			AccessModes:                   mode,
			PersistentVolumeReclaimPolicy: reclaim,
			StorageClassName:              scName,
		},
	}
}

func createPVC(name, namespace, cap string, mode []v1.PersistentVolumeAccessMode, class string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources:        v1.VolumeResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse(cap)}},
			AccessModes:      mode,
			StorageClassName: &class,
		},
	}
}

func createPVCWithNilStorageClass(name, namespace, cap string, mode []v1.PersistentVolumeAccessMode) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources:   v1.VolumeResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse(cap)}},
			AccessModes: mode,
		},
	}
}

// TestPersistentVolumeProvisionMultiPVCs tests provisioning of many PVCs.
// This test is configurable by KUBE_INTEGRATION_PV_* variables.
func TestPersistentVolumeProvisionMultiPVCs(t *testing.T) {
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount,StorageObjectInUseProtection"}, framework.SharedEtcd())
	defer s.TearDownFn()
	namespaceName := "provision-multi-pvs"

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	testClient, binder, informers, watchPV, watchPVC := createClients(tCtx, namespaceName, t, s, defaultSyncPeriod)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (PersistenceVolumes and StorageClasses).
	defer testClient.CoreV1().PersistentVolumes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	defer testClient.StorageV1().StorageClasses().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	go func() {
		w, err := testClient.EventsV1().Events(metav1.NamespaceAll).Watch(tCtx, metav1.ListOptions{})
		if err != nil {
			return
		}
		for {
			select {
			case event := <-w.ResultChan():
				writeToArtifacts(t.Name()+"-events.text", event.Object)
			case <-tCtx.Done():
				w.Stop()
				return
			}
		}
	}()

	storageClass := storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "gold",
		},
		Provisioner: provisionerPluginName,
	}
	testClient.StorageV1().StorageClasses().Create(context.TODO(), &storageClass, metav1.CreateOptions{})

	informers.Start(tCtx.Done())
	go binder.Run(tCtx)

	objCount := getObjectCount()
	pvcs := make([]*v1.PersistentVolumeClaim, objCount)
	for i := 0; i < objCount; i++ {
		pvc := createPVC("pvc-provision-"+strconv.Itoa(i), ns.Name, "1G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, "gold")
		pvcs[i] = pvc
	}

	klog.V(2).Infof("TestPersistentVolumeProvisionMultiPVCs: start")
	// Create the claims in a separate goroutine to pop events from watchPVC
	// early. It gets stuck with >3000 claims.
	go func() {
		for i := 0; i < objCount; i++ {
			_, _ = testClient.CoreV1().PersistentVolumeClaims(ns.Name).Create(context.TODO(), pvcs[i], metav1.CreateOptions{})
		}
	}()

	// Wait until the controller provisions and binds all of them
	for i := 0; i < objCount; i++ {
		waitForAnyPersistentVolumeClaimPhase1(t, watchPVC, v1.ClaimBound)
		klog.V(1).Infof("%d claims bound", i+1)
	}
	klog.V(2).Infof("TestPersistentVolumeProvisionMultiPVCs: claims are bound")

	// check that we have enough bound PVs
	pvList, err := testClient.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list volumes: %s", err)
	}
	if len(pvList.Items) != objCount {
		t.Fatalf("Expected to get %d volumes, got %d", objCount, len(pvList.Items))
	}
	for i := 0; i < objCount; i++ {
		pv := &pvList.Items[i]
		if pv.Status.Phase != v1.VolumeBound {
			t.Fatalf("Expected volume %s to be bound, is %s instead", pv.Name, pv.Status.Phase)
		}
		klog.V(2).Infof("PV %q is bound to PVC %q", pv.Name, pv.Spec.ClaimRef.Name)
	}

	// Delete the claims
	for i := 0; i < objCount; i++ {
		_ = testClient.CoreV1().PersistentVolumeClaims(ns.Name).Delete(context.TODO(), pvcs[i].Name, metav1.DeleteOptions{})
	}

	// Wait for the PVs to get deleted by listing remaining volumes
	// (delete events were unreliable)
	for {
		volumes, err := testClient.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			t.Fatalf("Failed to list volumes: %v", err)
		}

		klog.V(1).Infof("%d volumes remaining", len(volumes.Items))
		if len(volumes.Items) == 0 {
			break
		}
		time.Sleep(time.Second)
	}
	klog.V(2).Infof("TestPersistentVolumeProvisionMultiPVCs: volumes are deleted")
}

func waitForAnyPersistentVolumeClaimPhase1(t *testing.T, w watch.Interface, phase v1.PersistentVolumeClaimPhase) {
	for {
		event := <-w.ResultChan()
		writeToArtifacts(t.Name()+"watch-pvcs.text", event.Object)
		claim, ok := event.Object.(*v1.PersistentVolumeClaim)
		if !ok {
			continue
		}
		if claim.Status.Phase == phase {
			klog.V(2).Infof("claim %q is %s", claim.Name, phase)
			break
		}
	}
}
