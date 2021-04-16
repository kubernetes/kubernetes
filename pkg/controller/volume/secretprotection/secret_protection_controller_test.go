/*
Copyright 2021 The Kubernetes Authors.

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

package secretprotection

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/controller"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type reaction struct {
	verb      string
	resource  string
	reactorfn clienttesting.ReactionFunc
}

const (
	defaultNS         = "default"
	defaultPVName     = "pv1"
	defaultPVCName    = "pvc1"
	defaultSecretName = "secret1"
	defaultPodName    = "pod1"
	defaultNodeName   = "node1"
	defaultUID        = "uid1"
	defaultScName     = "sc1"
)

func pod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPodName,
			Namespace: defaultNS,
			UID:       defaultUID,
		},
		Spec: v1.PodSpec{
			NodeName: defaultNodeName,
		},
		Status: v1.PodStatus{
			Phase: v1.PodPending,
		},
	}
}

func unscheduled(pod *v1.Pod) *v1.Pod {
	pod.Spec.NodeName = ""
	return pod
}

func withSecret(secretName string, pod *v1.Pod) *v1.Pod {
	volume := v1.Volume{
		Name: secretName,
		VolumeSource: v1.VolumeSource{
			Secret: &v1.SecretVolumeSource{
				SecretName: secretName,
			},
		},
	}
	pod.Spec.Volumes = append(pod.Spec.Volumes, volume)
	return pod
}

func withEmptyDir(pod *v1.Pod) *v1.Pod {
	volume := v1.Volume{
		Name: "emptyDir",
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}
	pod.Spec.Volumes = append(pod.Spec.Volumes, volume)
	return pod
}

func withStatus(phase v1.PodPhase, pod *v1.Pod) *v1.Pod {
	pod.Status.Phase = phase
	return pod
}

func withUID(uid types.UID, pod *v1.Pod) *v1.Pod {
	pod.ObjectMeta.UID = uid
	return pod
}

func csiPV() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPVName,
			Namespace: defaultNS,
			UID:       defaultUID,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{},
			},
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func withControllerPublishSecret(secretName string, csiPV *v1.PersistentVolume) *v1.PersistentVolume {
	csiPV.Spec.PersistentVolumeSource.CSI.ControllerPublishSecretRef = &v1.SecretReference{Namespace: defaultNS, Name: secretName}
	return csiPV
}

func withNodeStageSecret(secretName string, csiPV *v1.PersistentVolume) *v1.PersistentVolume {
	csiPV.Spec.PersistentVolumeSource.CSI.NodeStageSecretRef = &v1.SecretReference{Namespace: defaultNS, Name: secretName}
	return csiPV
}

func withNodePublishSecret(secretName string, csiPV *v1.PersistentVolume) *v1.PersistentVolume {
	csiPV.Spec.PersistentVolumeSource.CSI.NodePublishSecretRef = &v1.SecretReference{Namespace: defaultNS, Name: secretName}
	return csiPV
}

func withControllerExpandSecret(secretName string, csiPV *v1.PersistentVolume) *v1.PersistentVolume {
	csiPV.Spec.PersistentVolumeSource.CSI.ControllerExpandSecretRef = &v1.SecretReference{Namespace: defaultNS, Name: secretName}
	return csiPV
}

func withVolumeStatus(phase v1.PersistentVolumePhase, pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.Status.Phase = phase
	return pv
}

func withPVUID(uid types.UID, pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.ObjectMeta.UID = uid
	return pv
}

func withScName(scName string, pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.Spec.StorageClassName = scName
	return pv
}

func bindWithPVC(pvcNamespace, pvcName string, pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.Spec.ClaimRef = &v1.ObjectReference{
		Namespace: pvcNamespace,
		Name:      pvcName,
	}
	return pv
}

func sc() *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultScName,
		},
		Parameters: map[string]string{},
	}
}

func addScParameters(params map[string]string, sc *storagev1.StorageClass) *storagev1.StorageClass {
	for k, v := range params {
		sc.Parameters[k] = v
	}
	return sc
}

func pvc() *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPVCName,
			Namespace: defaultNS,
		},
		Spec: v1.PersistentVolumeClaimSpec{},
	}
}

func secret() *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultSecretName,
			Namespace: defaultNS,
		},
	}
}

func withProtectionFinalizer(secret *v1.Secret) *v1.Secret {
	secret.Finalizers = append(secret.Finalizers, volumeutil.SecretProtectionFinalizer)
	return secret
}

func withNsAndName(ns, name string, secret *v1.Secret) *v1.Secret {
	secret.ObjectMeta.Namespace = ns
	secret.ObjectMeta.Name = name
	return secret
}

func deleted(secret *v1.Secret) *v1.Secret {
	secret.DeletionTimestamp = &metav1.Time{}
	return secret
}

func generateUpdateErrorFunc(t *testing.T, failures int) clienttesting.ReactionFunc {
	i := 0
	return func(action clienttesting.Action) (bool, runtime.Object, error) {
		i++
		if i <= failures {
			// Update fails
			update, ok := action.(clienttesting.UpdateAction)

			if !ok {
				t.Fatalf("Reactor got non-update action: %+v", action)
			}
			acc, _ := meta.Accessor(update.GetObject())
			return true, nil, apierrors.NewForbidden(update.GetResource().GroupResource(), acc.GetName(), errors.New("Mock error"))
		}
		// Update succeeds
		return false, nil, nil
	}
}

func TestSecretProtectionController(t *testing.T) {
	secretGVR := api.Resource("secrets").WithVersion("v1")
	podGVR := api.Resource("pods").WithVersion("v1")
	podGVK := api.Kind("Pod").WithVersion("v1")
	pvGVR := api.Resource("persistentvolumes").WithVersion("v1")
	pvGVK := api.Kind("PersistentVolume").WithVersion("v1")
	pvcGVR := api.Resource("persistentvolumeclaims").WithVersion("v1")
	scGVR := storageapi.Resource("storageclasses").WithVersion("v1")

	tests := []struct {
		name string
		// Object to insert into fake kubeclient before the test starts.
		initialObjects []runtime.Object
		// Whether not to insert the content of initialObjects into the
		// informers before the test starts. Set it to true to simulate the case
		// where informers have not been notified yet of certain API objects.
		informersAreLate bool
		// Optional client reactors.
		reactors []reaction
		// Secret event to simulate. This secret will be automatically added to
		// initialObjects.
		updatedSecret *v1.Secret
		// Pod event to simulate. This Pod will be automatically added to
		// initialObjects.
		updatedPod *v1.Pod
		// Pod event to simulate. This Pod is *not* added to
		// initialObjects.
		deletedPod *v1.Pod
		// PV event to simulate. This PV will be automatically added to
		// initialObjects.
		updatedPV *v1.PersistentVolume
		// PV event to simulate. This PV is *not* added to
		// initialObjects.
		deletedPV *v1.PersistentVolume
		// List of expected kubeclient actions that should happen during the
		// test.
		expectedActions                     []clienttesting.Action
		storageObjectInUseProtectionEnabled bool
	}{
		//
		// Secret events
		//
		{
			name:          "StorageObjectInUseProtection Enabled, Secret without finalizer -> finalizer is added",
			updatedSecret: secret(),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(secretGVR, defaultNS, withProtectionFinalizer(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:                                "StorageObjectInUseProtection Disabled, secret without finalizer -> finalizer is not added",
			updatedSecret:                       secret(),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: false,
		},
		{
			name:                                "secret with finalizer -> no action",
			updatedSecret:                       withProtectionFinalizer(secret()),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:          "saving secret finalizer fails -> controller retries",
			updatedSecret: secret(),
			reactors: []reaction{
				{
					verb:     "update",
					resource: "secrets",
					// update fails twice
					reactorfn: generateUpdateErrorFunc(t, 2),
				},
			},
			expectedActions: []clienttesting.Action{
				// This fails
				clienttesting.NewUpdateAction(secretGVR, defaultNS, withProtectionFinalizer(secret())),
				// This fails too
				clienttesting.NewUpdateAction(secretGVR, defaultNS, withProtectionFinalizer(secret())),
				// This succeeds
				clienttesting.NewUpdateAction(secretGVR, defaultNS, withProtectionFinalizer(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:          "StorageObjectInUseProtection Enabled, deleted secret with finalizer -> finalizer is removed",
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:          "StorageObjectInUseProtection Disabled, deleted secret with finalizer -> finalizer is removed",
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: false,
		},
		{
			name:          "finalizer removal fails -> controller retries",
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			reactors: []reaction{
				{
					verb:     "update",
					resource: "secrets",
					// update fails twice
					reactorfn: generateUpdateErrorFunc(t, 2),
				},
			},
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				// Fails
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				// Fails too
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				// Succeeds
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		// blocked or not blocked related to references from pod
		{
			name: "deleted secret with finalizer + pod with the secret exists -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withSecret(defaultSecretName, pod()),
			},
			updatedSecret:   deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "deleted secret with finalizer + pod with unrelated secret and EmptyDir exists -> finalizer is removed",
			initialObjects: []runtime.Object{
				withEmptyDir(withSecret("unrelatedSecret", pod())),
			},
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + pod with the secret finished but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withStatus(v1.PodFailed, withSecret(defaultSecretName, pod())),
			},
			updatedSecret:                       deleted(withProtectionFinalizer(secret())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + pod with the secret exists but is not in the Informer's cache yet -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withSecret(defaultSecretName, pod()),
			},
			informersAreLate: true,
			updatedSecret:    deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		// blocked or not blocked related to references from PV
		{
			name: "deleted secret with finalizer + CSI PV with the controller publish secret exists -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withControllerPublishSecret(defaultSecretName, csiPV()),
			},
			updatedSecret:   deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "deleted secret with finalizer + CSI PV with the node stage secret exists -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withNodeStageSecret(defaultSecretName, csiPV()),
			},
			updatedSecret:   deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "deleted secret with finalizer + CSI PV with the node publish secret exists -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withNodePublishSecret(defaultSecretName, csiPV()),
			},
			updatedSecret:   deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "deleted secret with finalizer + CSI PV with unrelated secret set to node publish secret -> finalizer is removed",
			initialObjects: []runtime.Object{
				withNodePublishSecret("UnrelatedSecret", csiPV()),
			},
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV with the node stage secret is requested to delete but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withVolumeStatus(v1.VolumeReleased, withNodeStageSecret(defaultSecretName, csiPV())),
			},
			updatedSecret:                       deleted(withProtectionFinalizer(secret())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV referencing the StorageClass that directly referencing the secret as provisioner secret is requested to delete but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withVolumeStatus(v1.VolumeReleased, withScName(defaultScName, csiPV())),
				addScParameters(
					map[string]string{
						"provisioner-secret-namespace":               defaultNS,
						"csi.storage.k8s.io/provisioner-secret-name": defaultSecretName,
					}, sc()),
			},
			updatedSecret:                       deleted(withProtectionFinalizer(secret())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV referencing the StorageClass that directly referencing UNRELATED secret as provisioner secret is requested to delete but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withVolumeStatus(v1.VolumeReleased, withScName(defaultScName, csiPV())),
				addScParameters(
					map[string]string{
						"provisioner-secret-namespace":               defaultNS,
						"csi.storage.k8s.io/provisioner-secret-name": "unrelated-secret",
					}, sc()),
			},
			updatedSecret: deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewRootGetAction(scGVR, defaultScName),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV referencing the StorageClass that referencing the secret as provisioner secret via PVC information is requested to delete but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withVolumeStatus(v1.VolumeReleased, bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV()))),
				pvc(),
				addScParameters(
					map[string]string{
						"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}",
						"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
					}, sc()),
			},
			updatedSecret: deleted(withProtectionFinalizer(
				withNsAndName(
					fmt.Sprintf("ns-%s-%s", defaultPVName, defaultNS),
					fmt.Sprintf("sec-%s-%s-%s", defaultPVName, defaultNS, defaultPVCName),
					secret()))),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV referencing the StorageClass that referencing the UNRELATED secret as provisioner secret via PVC information is requested to delete but is not deleted -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withVolumeStatus(v1.VolumeReleased, bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV()))),
				pvc(),
				addScParameters(
					map[string]string{
						"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}",
						"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
					}, sc()),
			},
			updatedSecret: deleted(withProtectionFinalizer(
				withNsAndName(
					fmt.Sprintf("ns-%s-%s", defaultPVName, defaultNS),
					fmt.Sprintf("unrelated-sec-%s-%s-%s", defaultPVName, defaultNS, defaultPVCName),
					secret()))),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK,
					fmt.Sprintf("ns-%s-%s", defaultPVName, defaultNS),
					metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewRootGetAction(scGVR, defaultScName),
				clienttesting.NewGetAction(pvcGVR, defaultNS, defaultPVCName),
				clienttesting.NewUpdateAction(secretGVR,
					fmt.Sprintf("ns-%s-%s", defaultPVName, defaultNS),
					deleted(
						withNsAndName(
							fmt.Sprintf("ns-%s-%s", defaultPVName, defaultNS),
							fmt.Sprintf("unrelated-sec-%s-%s-%s", defaultPVName, defaultNS, defaultPVCName),
							secret()))),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV with the controller publish secret exists but is not in the Informer's cache yet -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withControllerPublishSecret(defaultSecretName, csiPV()),
			},
			informersAreLate: true,
			updatedSecret:    deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted secret with finalizer + CSI PV with the controller expand secret exists but is not in the Informer's cache yet -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withControllerExpandSecret(defaultSecretName, csiPV()),
			},
			informersAreLate: true,
			updatedSecret:    deleted(withProtectionFinalizer(secret())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		//
		// Pod events
		//
		{
			name: "updated running Pod -> no action",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			updatedPod:                          withStatus(v1.PodRunning, withSecret(defaultSecretName, pod())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "updated finished Pod -> finalizer is not removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			updatedPod:                          withStatus(v1.PodSucceeded, withSecret(defaultSecretName, pod())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "updated unscheduled Pod -> finalizer is removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			updatedPod: unscheduled(withSecret(defaultSecretName, pod())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "deleted running Pod -> finalizer is removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPod: withStatus(v1.PodRunning, withSecret(defaultSecretName, pod())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "pod delete and create with same namespaced name seen as an update, old pod used deleted secret -> finalizer is removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPod: withSecret(defaultSecretName, pod()),
			updatedPod: withUID("uid2", pod()),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "pod delete and create with same namespaced name seen as an update, old pod used non-deleted secret -> finalizer is not removed",
			initialObjects: []runtime.Object{
				// Not deleted
				withProtectionFinalizer(secret()),
			},
			deletedPod:                          withSecret(defaultSecretName, pod()),
			updatedPod:                          withUID("uid2", pod()),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "pod delete and create with same namespaced name seen as an update, both pods reference deleted secret -> finalizer is not removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPod:                          withSecret(defaultSecretName, pod()),
			updatedPod:                          withUID("uid2", withSecret(defaultSecretName, pod())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "pod update from unscheduled to scheduled, deleted secret is referenced -> finalizer is not removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPod:                          unscheduled(withSecret(defaultSecretName, pod())),
			updatedPod:                          withSecret(defaultSecretName, pod()),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		//
		// PV events
		//
		{
			name: "updated PV status -> no action",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			updatedPV:                           withVolumeStatus(v1.VolumeBound, withNodeStageSecret(defaultSecretName, csiPV())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "PV delete and create seen as an update, old PV used deleted secret -> finalizer is removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPV: withNodeStageSecret(defaultSecretName, csiPV()),
			updatedPV: withPVUID("uid2", csiPV()),
			expectedActions: []clienttesting.Action{
				clienttesting.NewListAction(podGVR, podGVK, defaultNS, metav1.ListOptions{}),
				clienttesting.NewRootListAction(pvGVR, pvGVK, metav1.ListOptions{}),
				clienttesting.NewUpdateAction(secretGVR, defaultNS, deleted(secret())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "PV delete and create seen as an update, old PV used non-deleted secret -> finalizer is not removed",
			initialObjects: []runtime.Object{
				// Not deleted
				withProtectionFinalizer(secret()),
			},
			deletedPV:                           withNodeStageSecret(defaultSecretName, csiPV()),
			updatedPV:                           withPVUID("uid2", csiPV()),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name: "PV delete and create seen as an update, both PVs reference deleted secret -> finalizer is not removed",
			initialObjects: []runtime.Object{
				deleted(withProtectionFinalizer(secret())),
			},
			deletedPV:                           withNodePublishSecret(defaultSecretName, csiPV()),
			updatedPV:                           withPVUID("uid2", withControllerPublishSecret(defaultSecretName, csiPV())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
	}

	for _, test := range tests {
		// Create initial data for client and informers.
		var (
			clientObjs    []runtime.Object
			informersObjs []runtime.Object
		)
		if test.updatedSecret != nil {
			clientObjs = append(clientObjs, test.updatedSecret)
			informersObjs = append(informersObjs, test.updatedSecret)
		}
		if test.updatedPod != nil {
			clientObjs = append(clientObjs, test.updatedPod)
			informersObjs = append(informersObjs, test.updatedPod)
		}
		if test.updatedPV != nil {
			clientObjs = append(clientObjs, test.updatedPV)
			informersObjs = append(informersObjs, test.updatedPV)
		}
		clientObjs = append(clientObjs, test.initialObjects...)
		if !test.informersAreLate {
			informersObjs = append(informersObjs, test.initialObjects...)
		}

		// Create client with initial data
		client := fake.NewSimpleClientset(clientObjs...)

		// Create informers
		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		secretInformer := informers.Core().V1().Secrets()
		podInformer := informers.Core().V1().Pods()
		pvInformer := informers.Core().V1().PersistentVolumes()
		pvcInformer := informers.Core().V1().PersistentVolumeClaims()
		scInformer := informers.Storage().V1().StorageClasses()

		// Create the controller
		ctrl := NewSecretProtectionController(secretInformer, podInformer, pvInformer, pvcInformer, scInformer, client, test.storageObjectInUseProtectionEnabled)

		// Populate the informers with initial objects so the controller can
		// Get() and List() it.
		for _, obj := range informersObjs {
			switch obj.(type) {
			case *v1.Secret:
				secretInformer.Informer().GetStore().Add(obj)
			case *v1.Pod:
				podInformer.Informer().GetStore().Add(obj)
			case *v1.PersistentVolume:
				pvInformer.Informer().GetStore().Add(obj)
			case *v1.PersistentVolumeClaim:
				pvcInformer.Informer().GetStore().Add(obj)
			case *storagev1.StorageClass:
				scInformer.Informer().GetStore().Add(obj)
			default:
				t.Fatalf("Unknown initalObject type: %+v", obj)
			}
		}

		// Add reactor to inject test errors.
		for _, reactor := range test.reactors {
			client.Fake.PrependReactor(reactor.verb, reactor.resource, reactor.reactorfn)
		}

		// Start the test by simulating an event
		if test.updatedSecret != nil {
			ctrl.secretAddedUpdated(test.updatedSecret)
		}
		switch {
		case test.deletedPod != nil && test.updatedPod != nil && test.deletedPod.Namespace == test.updatedPod.Namespace && test.deletedPod.Name == test.updatedPod.Name:
			ctrl.podAddedDeletedUpdated(test.deletedPod, test.updatedPod, false)
		case test.updatedPod != nil:
			ctrl.podAddedDeletedUpdated(nil, test.updatedPod, false)
		case test.deletedPod != nil:
			ctrl.podAddedDeletedUpdated(nil, test.deletedPod, true)
		case test.deletedPV != nil && test.updatedPV != nil && test.deletedPV.Name == test.updatedPV.Name:
			ctrl.pvAddedDeletedUpdated(test.deletedPV, test.updatedPV, false)
		case test.updatedPV != nil:
			ctrl.pvAddedDeletedUpdated(nil, test.updatedPV, false)
		case test.deletedPV != nil:
			ctrl.pvAddedDeletedUpdated(nil, test.deletedPV, true)
		}

		// Process the controller queue until we get expected results
		timeout := time.Now().Add(10 * time.Second)
		lastReportedActionCount := 0
		for {
			if time.Now().After(timeout) {
				t.Errorf("Test %q: timed out", test.name)
				break
			}
			if ctrl.queue.Len() > 0 {
				klog.V(5).Infof("Test %q: %d events queue, processing one", test.name, ctrl.queue.Len())
				ctrl.processNextWorkItem()
			}
			if ctrl.queue.Len() > 0 {
				// There is still some work in the queue, process it now
				continue
			}
			currentActionCount := len(client.Actions())
			if currentActionCount < len(test.expectedActions) {
				// Do not log every wait, only when the action count changes.
				if lastReportedActionCount < currentActionCount {
					klog.V(5).Infof("Test %q: got %d actions out of %d, waiting for the rest", test.name, currentActionCount, len(test.expectedActions))
					lastReportedActionCount = currentActionCount
				}
				// The test expected more to happen, wait for the actions.
				// Most probably it's exponential backoff
				time.Sleep(10 * time.Millisecond)
				continue
			}
			break
		}
		actions := client.Actions()
		for i, action := range actions {
			if len(test.expectedActions) < i+1 {
				t.Errorf("Test %q: %d unexpected actions: %+v", test.name, len(actions)-len(test.expectedActions), spew.Sdump(actions[i:]))
				break
			}

			expectedAction := test.expectedActions[i]
			if !reflect.DeepEqual(expectedAction, action) {
				t.Errorf("Test %q: action %d\nExpected:\n%s\ngot:\n%s", test.name, i, spew.Sdump(expectedAction), spew.Sdump(action))
			}
		}

		if len(test.expectedActions) > len(actions) {
			t.Errorf("Test %q: %d additional expected actions", test.name, len(test.expectedActions)-len(actions))
			for _, a := range test.expectedActions[len(actions):] {
				t.Logf("    %+v", a)
			}
		}

	}
}

func TestGetProvisionerSecretKey(t *testing.T) {
	tests := []struct {
		name      string
		pv        *v1.PersistentVolume
		pvc       *v1.PersistentVolumeClaim
		sc        *storagev1.StorageClass
		askAPI    bool
		expected  string
		expectErr bool
	}{
		// Informer cases
		{
			name:      "PV without StorageClass name returns empty and no error",
			pv:        csiPV(),
			expected:  "",
			expectErr: false,
		},
		{
			name:      "PV with non-existent StorageClass name returns empty and error",
			pv:        withScName("non-existent-sc", csiPV()),
			sc:        sc(),
			expected:  "",
			expectErr: true,
		},
		{
			name:      "PV with StorageClass name that has no parameters returns empty and no error",
			pv:        withScName(defaultScName, csiPV()),
			sc:        sc(),
			expected:  "",
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has deprecated provisioner key parameters w/o replacement returns the expected key and no error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"unrelated":                    "should have no effect",
					"provisioner-secret-namespace": "ns1",
					"provisioner-secret-name":      "sec1",
				}, sc()),
			expected:  "ns1/sec1",
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/o replacement returns the expected key and no error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns1",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec1",
				}, sc()),
			expected:  "ns1/sec1",
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has mixed(deprecated/current) provisioner key parameters w/o replacement returns the expected key and no error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"provisioner-secret-namespace":               "ns1",
					"csi.storage.k8s.io/provisioner-secret-name": "sec1",
				}, sc()),
			expected:  "ns1/sec1",
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameter only for ns returns empty and error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"provisioner-secret-namespace": "ns1",
				}, sc()),
			expected:  "",
			expectErr: true,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameter only for name returns empty and error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"csi.storage.k8s.io/provisioner-secret-name": "sec1",
				}, sc()),
			expected:  "",
			expectErr: true,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/ pv related replacement returns the expected key and no error",
			pv:   withScName(defaultScName, csiPV()),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}",
				}, sc()),
			expected:  fmt.Sprintf("ns-%s/sec-%s", defaultPVName, defaultPVName),
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/ pvc related replacement returns the expected key and no error",
			pv:   bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV())),
			pvc:  pvc(),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pvc.namespace}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pvc.name}",
				}, sc()),
			expected:  fmt.Sprintf("ns-%s/sec-%s", defaultNS, defaultPVCName),
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/ multiple replacement returns the expected key and no error",
			pv:   bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV())),
			pvc:  pvc(),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
				}, sc()),
			expected: fmt.Sprintf("ns-%s-%s/sec-%s-%s-%s",
				defaultPVName, defaultNS,
				defaultPVName, defaultNS, defaultPVCName),
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/ invalid replacement token returns empty and error",
			pv:   bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV())),
			pvc:  pvc(),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}-${pvc.name}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
				}, sc()),
			expected:  "",
			expectErr: true,
		},
		// API cases
		{
			name:      "With askAPI true, PV with non-existent StorageClass name returns empty and error",
			pv:        withScName("non-existent-sc", csiPV()),
			sc:        sc(),
			askAPI:    true,
			expected:  "",
			expectErr: true,
		},
		{
			name:      "With askAPI true, PV with StorageClass name that has no parameters returns empty and no error",
			pv:        withScName(defaultScName, csiPV()),
			sc:        sc(),
			askAPI:    true,
			expected:  "",
			expectErr: false,
		},
		{
			name: "With askAPI true, PV with StorageClass name that has provisioner key parameters w/ multiple replacement returns the expected key and no error",
			pv:   bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV())),
			pvc:  pvc(),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
				}, sc()),
			askAPI: true,
			expected: fmt.Sprintf("ns-%s-%s/sec-%s-%s-%s",
				defaultPVName, defaultNS,
				defaultPVName, defaultNS, defaultPVCName),
			expectErr: false,
		},
		{
			name: "PV with StorageClass name that has provisioner key parameters w/ invalid replacement token returns empty and error",
			pv:   bindWithPVC(defaultNS, defaultPVCName, withScName(defaultScName, csiPV())),
			pvc:  pvc(),
			sc: addScParameters(
				map[string]string{
					"unrelated": "should have no effect",
					"csi.storage.k8s.io/provisioner-secret-namespace": "ns-${pv.name}-${pvc.namespace}-${pvc.name}",
					"csi.storage.k8s.io/provisioner-secret-name":      "sec-${pv.name}-${pvc.namespace}-${pvc.name}",
				}, sc()),
			askAPI:    true,
			expected:  "",
			expectErr: true,
		},
	}

	for _, test := range tests {
		var (
			clientObjs    []runtime.Object
			informersObjs []runtime.Object
		)
		if test.pv != nil {
			clientObjs = append(clientObjs, test.pv)
			informersObjs = append(informersObjs, test.pv)
		}
		if test.pvc != nil {
			clientObjs = append(clientObjs, test.pvc)
			informersObjs = append(informersObjs, test.pvc)
		}
		if test.sc != nil {
			clientObjs = append(clientObjs, test.sc)
			informersObjs = append(informersObjs, test.sc)
		}

		// Create client with initial data
		client := fake.NewSimpleClientset(clientObjs...)

		// Create informers
		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		secretInformer := informers.Core().V1().Secrets()
		podInformer := informers.Core().V1().Pods()
		pvInformer := informers.Core().V1().PersistentVolumes()
		pvcInformer := informers.Core().V1().PersistentVolumeClaims()
		scInformer := informers.Storage().V1().StorageClasses()

		// Create the controller
		ctrl := NewSecretProtectionController(secretInformer, podInformer, pvInformer, pvcInformer, scInformer, client, true /* storageObjectInUseProtectionEnabled */)

		// Populate the informers with initial objects so the controller can
		// Get() and List() it.
		for _, obj := range informersObjs {
			switch obj.(type) {
			case *v1.Secret:
				secretInformer.Informer().GetStore().Add(obj)
			case *v1.Pod:
				podInformer.Informer().GetStore().Add(obj)
			case *v1.PersistentVolume:
				pvInformer.Informer().GetStore().Add(obj)
			case *v1.PersistentVolumeClaim:
				pvcInformer.Informer().GetStore().Add(obj)
			case *storagev1.StorageClass:
				scInformer.Informer().GetStore().Add(obj)
			default:
				t.Fatalf("Unknown initalObject type: %+v", obj)
			}
		}

		result, err := ctrl.getProvisionerSecretKey(test.pv, test.askAPI)

		if test.expectErr {
			if err == nil {
				t.Errorf("Test %q: expects error but got no error", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("Test %q: expects no error but got error: %v", test.name, err)
			}
		}

		if result != test.expected {
			t.Errorf("Test %q: expects %q but got %q", test.name, test.expected, result)
		}
	}
}
