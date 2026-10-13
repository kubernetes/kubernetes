/*
Copyright The Kubernetes Authors.

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

package volume

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientgoinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	csiplugin "k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const (
	csiDriverName   = "csi-test-driver"
	csiVolumeHandle = "test-volume-handle"
	// testNodeName aligns with fakePodWithPVC.
	testNodeName     = "node-sandbox"
	stalePVFinalizer = "integration.test.k8s.io/keep-terminating"
)

// csiTestPV builds a PersistentVolume for the volume handle the tests share. Only the name
// differs between the PVs they build, so they are all the same underlying volume and map to
// one VolumeAttachment name on a node.
func csiTestPV(name string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("5Gi"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       csiDriverName,
					VolumeHandle: csiVolumeHandle,
				},
			},
		},
	}
}

// TestVolumeAttachmentOfTerminatingPVIsReplaced covers a workload whose PersistentVolume is
// recreated under a new name for the same volume, as statically provisioned PVs are when the
// workload is restarted. Two PVs then map to one VolumeAttachment, because its name is derived
// from the volume handle and not from the PV.
//
// The attach of the first PV is still queued when that PV is deleted. external-attacher
// refuses it from then on, so the VolumeAttachment created for it can never attach, and its
// spec is immutable, so the controller has to delete it and create it again for the PV of the
// new pod.
func TestVolumeAttachmentOfTerminatingPVIsReplaced(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")

	testClient, ctrl, informers := createCSIAdClients(tCtx, t, server)

	ns := framework.CreateNamespaceOrDie(testClient, "test-previous-volumeattachment", t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation: "true",
			},
		},
	}
	if _, err := testClient.CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{}); err != nil {
		tCtx.Fatalf("Failed to create node: %v", err)
	}

	// The workload as it runs before it is restarted.
	previousPV, previousPVC := createBoundCSIPVAndPVC(tCtx, testClient, ns.Name, "test-pv-previous", "test-pvc-previous")
	previousPod := createCSITestPod(tCtx, testClient, ns.Name, "test-pod-previous", previousPVC.Name)

	informers.Start(tCtx.Done())
	informers.WaitForCacheSync(tCtx.Done())
	go ctrl.Run(tCtx)

	// The controller creates a VolumeAttachment for the PV of that pod and waits for
	// external-attacher to attach it.
	previousAttachment := waitForVolumeAttachment(tCtx, testClient, previousPV.Name)

	// The workload is restarted while the volume is still not attached: its PV is recreated
	// under a new name for the same volume handle, and its pod is replaced by one that
	// arrives before it is gone. The volume therefore never leaves the desired state of
	// world, so the controller has to replace the PersistentVolume recorded for it rather
	// than record the new one when the volume is added.
	currentPV, currentPVC := createBoundCSIPVAndPVC(tCtx, testClient, ns.Name, "test-pv-current", "test-pvc-current")
	currentPod := createCSITestPod(tCtx, testClient, ns.Name, "test-pod-current", currentPVC.Name)
	if err := testClient.CoreV1().Pods(ns.Name).Delete(tCtx, previousPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
		tCtx.Fatalf("Failed to delete the pod of the previous generation: %v", err)
	}
	terminatePV(tCtx, testClient, previousPV.Name)

	// external-attacher refuses to attach a PV that is being deleted. This also ends the
	// wait of the attach that is in flight, which would otherwise hold the volume for the
	// CSI timeout.
	refuseAttachment(tCtx, testClient, previousAttachment, previousPV.Name)

	// The controller is expected to record the PV of the new pod, to replace the
	// VolumeAttachment it created for the PV that is now being deleted, and to attach the
	// volume for the pod that is left.
	newAttachment := waitForVolumeAttachment(tCtx, testClient, currentPV.Name)
	attachVolume(tCtx, testClient, newAttachment)
	waitForVolumeToBeAttached(tCtx, t, testClient, currentPod.Name, testNodeName)
}

// createBoundCSIPVAndPVC creates a PersistentVolume for the volume handle the tests share
// and a PersistentVolumeClaim bound to it, as the PV controller would leave them.
func createBoundCSIPVAndPVC(tCtx ktesting.TContext, client clientset.Interface, namespace, pvName, pvcName string) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	tCtx.Helper()

	pv := csiTestPV(pvName)
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      pvcName,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
			VolumeName:       pv.Name,
			StorageClassName: new(""),
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: resource.MustParse("5Gi"),
				},
			},
		},
	}
	pvc, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(tCtx, pvc, metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to create PVC %s: %v", pvcName, err)
	}
	pv.Spec.ClaimRef = &v1.ObjectReference{
		Kind:      "PersistentVolumeClaim",
		Namespace: namespace,
		Name:      pvc.Name,
		UID:       pvc.UID,
	}
	pv, err = client.CoreV1().PersistentVolumes().Create(tCtx, pv, metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to create PV %s: %v", pvName, err)
	}
	pvc.Status.Phase = v1.ClaimBound
	pvc, err = client.CoreV1().PersistentVolumeClaims(namespace).UpdateStatus(tCtx, pvc, metav1.UpdateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to bind PVC %s: %v", pvcName, err)
	}
	return pv, pvc
}

// createCSITestPod creates a pod that is scheduled to the test node and uses the given
// claim.
func createCSITestPod(tCtx ktesting.TContext, client clientset.Interface, namespace, name, pvcName string) *v1.Pod {
	tCtx.Helper()

	// The claim of the fixture is not used, the tests create and bind their own.
	pod, _ := fakePodWithPVC(name, pvcName, namespace)
	pod, err := client.CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to create pod %s: %v", name, err)
	}
	return pod
}

// terminatePV deletes the PersistentVolume but holds it in Terminating, the way the
// external-attacher finalizer does while a VolumeAttachment still refers to it.
func terminatePV(tCtx ktesting.TContext, client clientset.Interface, pvName string) {
	tCtx.Helper()

	pv, err := client.CoreV1().PersistentVolumes().Get(tCtx, pvName, metav1.GetOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to get PV %s: %v", pvName, err)
	}
	pv.Finalizers = append(pv.Finalizers, stalePVFinalizer)
	if _, err := client.CoreV1().PersistentVolumes().Update(tCtx, pv, metav1.UpdateOptions{}); err != nil {
		tCtx.Fatalf("Failed to add a finalizer to PV %s: %v", pvName, err)
	}
	if err := client.CoreV1().PersistentVolumes().Delete(tCtx, pvName, metav1.DeleteOptions{}); err != nil {
		tCtx.Fatalf("Failed to delete PV %s: %v", pvName, err)
	}
}

func refuseAttachment(tCtx ktesting.TContext, client clientset.Interface, attachment *storagev1.VolumeAttachment, pvName string) {
	tCtx.Helper()

	attachment.Status.AttachError = &storagev1.VolumeError{
		Time:    metav1.Now(),
		Message: fmt.Sprintf("PersistentVolume %q is marked for deletion", pvName),
	}
	if _, err := client.StorageV1().VolumeAttachments().UpdateStatus(tCtx, attachment, metav1.UpdateOptions{}); err != nil {
		tCtx.Fatalf("Failed to set the attach error on VolumeAttachment %s: %v", attachment.Name, err)
	}
}

// waitForVolumeAttachment waits for the VolumeAttachment of the volume handle the tests
// share to refer to the named PersistentVolume. Its spec is immutable, so an object that
// refers to that PV is one that was created for it.
func waitForVolumeAttachment(tCtx ktesting.TContext, client clientset.Interface, pvName string) *storagev1.VolumeAttachment {
	tCtx.Helper()

	attachmentName := csiplugin.GetVolumeAttachmentName(csiVolumeHandle, csiDriverName, testNodeName)
	var attachment, lastSeen *storagev1.VolumeAttachment
	err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true,
		func(ctx context.Context) (bool, error) {
			found, err := client.StorageV1().VolumeAttachments().Get(ctx, attachmentName, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				lastSeen = nil
				return false, nil
			}
			if err != nil {
				return false, err
			}
			lastSeen = found
			if ptr.Deref(found.Spec.Source.PersistentVolumeName, "") != pvName {
				return false, nil
			}
			attachment = found
			return true, nil
		})
	if err != nil {
		tCtx.Fatalf("No VolumeAttachment %s referring to PV %s: %v, last seen: %+v",
			attachmentName, pvName, err, lastSeen)
	}
	return attachment
}

// attachVolume marks the VolumeAttachment attached, the way external-attacher would.
// Which volume the node reports as attached does not tell the PersistentVolumes of one
// volume handle apart, they all map to one unique volume name, so it is the source of the
// VolumeAttachment that the tests assert on.
func attachVolume(tCtx ktesting.TContext, client clientset.Interface, attachment *storagev1.VolumeAttachment) {
	tCtx.Helper()

	attachment.Status.Attached = true
	if _, err := client.StorageV1().VolumeAttachments().UpdateStatus(tCtx, attachment, metav1.UpdateOptions{}); err != nil {
		tCtx.Fatalf("Failed to mark VolumeAttachment %s attached: %v", attachment.Name, err)
	}
}

// createCSIAdClients creates an attach/detach controller that runs the CSI volume plugin.
func createCSIAdClients(ctx context.Context, t *testing.T, server *kubeapiservertesting.TestServer) (*clientset.Clientset, attachdetach.AttachDetachController, clientgoinformers.SharedInformerFactory) {
	testClient := clientset.NewForConfigOrDie(server.ClientConfig)
	informers := clientgoinformers.NewSharedInformerFactory(testClient, 12*time.Hour)

	ctrl, err := attachdetach.NewAttachDetachController(
		ctx,
		testClient,
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Core().V1().PersistentVolumes(),
		informers.Storage().V1().CSINodes(),
		informers.Storage().V1().CSIDrivers(),
		informers.Storage().V1().VolumeAttachments(),
		csiplugin.ProbeVolumePlugins(),
		nil, /* prober */
		false,
		5*time.Second,
		false,
		defaultTimerConfig,
	)
	if err != nil {
		t.Fatalf("Error creating AttachDetachController: %v", err)
	}
	return testClient, ctrl, informers
}
