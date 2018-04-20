/*
Copyright 2018 The Kubernetes Authors.

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

package storageprotection

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)


type reaction struct {
	verb      string
	resource  string
	reactorfn clienttesting.ReactionFunc
}

const (
	defaultNS       = "default"
	defaultPVCName  = "pvc1"
	defaultPodName  = "pod1"
	defaultNodeName = "node1"
	defaultPVName   = "default-pv"
)

func pod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPodName,
			Namespace: defaultNS,
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

func withPVC(pvcName string, pod *v1.Pod) *v1.Pod {
	volume := v1.Volume{
		Name: pvcName,
		VolumeSource: v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
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

func pvc() *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPVCName,
			Namespace: defaultNS,
		},
	}
}

func withPVCProtectionFinalizer(pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {
	pvc.Finalizers = append(pvc.Finalizers, volumeutil.PVCProtectionFinalizer)
	return pvc
}

func pvcDeleted(pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {
	pvc.DeletionTimestamp = &metav1.Time{}
	return pvc
}

func pv() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultPVName,
		},
	}
}

func boundPV() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultPVName,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeBound,
		},
	}
}

func withPVProtectionFinalizer(pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.Finalizers = append(pv.Finalizers, volumeutil.PVProtectionFinalizer)
	return pv
}

func pvDeleted(pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.DeletionTimestamp = &metav1.Time{}
	return pv
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
			return true, nil, apierrors.NewForbidden(update.GetResource().GroupResource(), acc.GetName(), errors.New("mock error"))
		}
		// Update succeeds
		return false, nil, nil
	}
}

func TestPVCProtectionController(t *testing.T) {
	pvcVer := schema.GroupVersionResource{
		Group:    v1.GroupName,
		Version:  "v1",
		Resource: "persistentvolumeclaims",
	}
	pvVer := schema.GroupVersionResource{
		Group:    v1.GroupName,
		Version:  "v1",
		Resource: "persistentvolumes",
	}

	tests := []struct {
		name string
		// Object to insert into fake kubeclient before the test starts.
		initialObjects []runtime.Object
		// Optional client reactors.
		reactors []reaction
		// PV event to simulate. This PV will be automatically added to
		// initalObjects.
		updatedPV *v1.PersistentVolume
		// PVC event to simulate. This PVC will be automatically added to
		// initalObjects.
		updatedPVC *v1.PersistentVolumeClaim
		// Pod event to simulate. This Pod will be automatically added to
		// initalObjects.
		updatedPod *v1.Pod
		// Pod event to similate. This Pod is *not* added to
		// initalObjects.
		deletedPod *v1.Pod
		// List of expected kubeclient actions that should happen during the
		// test.
		expectedActions []clienttesting.Action
	}{
		//
		// PV events
		//
		{
			name:      "PV without finalizer -> finalizer is added",
			updatedPV: pv(),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvVer, "", withPVProtectionFinalizer(pv())),
			},
		},
		{
			name:            "PVC with finalizer -> no action",
			updatedPV:       withPVProtectionFinalizer(pv()),
			expectedActions: []clienttesting.Action{},
		},
		{
			name:      "saving PVC finalizer fails -> controller retries",
			updatedPV: pv(),
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "persistentvolumes",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// This fails
				clienttesting.NewUpdateAction(pvVer, "", withPVProtectionFinalizer(pv())),
				// This fails too
				clienttesting.NewUpdateAction(pvVer, "", withPVProtectionFinalizer(pv())),
				// This succeeds
				clienttesting.NewUpdateAction(pvVer, "", withPVProtectionFinalizer(pv())),
			},
		},
		{
			name:      "deleted PV with finalizer -> finalizer is removed",
			updatedPV: pvDeleted(withPVProtectionFinalizer(pv())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvVer, "", pvDeleted(pv())),
			},
		},
		{
			name:      "finalizer removal fails -> controller retries",
			updatedPV: pvDeleted(withPVProtectionFinalizer(pv())),
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "persistentvolumes",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// Fails
				clienttesting.NewUpdateAction(pvVer, "", pvDeleted(pv())),
				// Fails too
				clienttesting.NewUpdateAction(pvVer, "", pvDeleted(pv())),
				// Succeeds
				clienttesting.NewUpdateAction(pvVer, "", pvDeleted(pv())),
			},
		},
		{
			name:            "deleted PVC with finalizer + PV is bound -> finalizer is not removed",
			updatedPV:       pvDeleted(withPVProtectionFinalizer(boundPV())),
			expectedActions: []clienttesting.Action{},
		},
		// PVC events
		//
		{
			name:       "PVC without finalizer -> finalizer is added",
			updatedPVC: pvc(),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, withPVCProtectionFinalizer(pvc())),
			},
		},
		{
			name:            "PVC with finalizer -> no action",
			updatedPVC:      withPVCProtectionFinalizer(pvc()),
			expectedActions: []clienttesting.Action{},
		},
		{
			name:       "saving PVC finalizer fails -> controller retries",
			updatedPVC: pvc(),
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "persistentvolumeclaims",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// This fails
				clienttesting.NewUpdateAction(pvcVer, defaultNS, withPVCProtectionFinalizer(pvc())),
				// This fails too
				clienttesting.NewUpdateAction(pvcVer, defaultNS, withPVCProtectionFinalizer(pvc())),
				// This succeeds
				clienttesting.NewUpdateAction(pvcVer, defaultNS, withPVCProtectionFinalizer(pvc())),
			},
		},
		{
			name:       "deleted PVC with finalizer -> finalizer is removed",
			updatedPVC: pvcDeleted(withPVCProtectionFinalizer(pvc())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		{
			name:       "finalizer removal fails -> controller retries",
			updatedPVC: pvcDeleted(withPVCProtectionFinalizer(pvc())),
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "persistentvolumeclaims",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// Fails
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
				// Fails too
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
				// Succeeds
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		{
			name: "deleted PVC with finalizer + pods with the PVC exists -> finalizer is not removed",
			initialObjects: []runtime.Object{
				withPVC(defaultPVCName, pod()),
			},
			updatedPVC:      pvcDeleted(withPVCProtectionFinalizer(pvc())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "deleted PVC with finalizer + pods with unrelated PVC and EmptyDir exists -> finalizer is removed",
			initialObjects: []runtime.Object{
				withEmptyDir(withPVC("unrelatedPVC", pod())),
			},
			updatedPVC: pvcDeleted(withPVCProtectionFinalizer(pvc())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		{
			name: "deleted PVC with finalizer + pods with the PVC andis finished -> finalizer is removed",
			initialObjects: []runtime.Object{
				withStatus(v1.PodFailed, withPVC(defaultPVCName, pod())),
			},
			updatedPVC: pvcDeleted(withPVCProtectionFinalizer(pvc())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		//
		// Pod events
		//
		{
			name: "updated running Pod -> no action",
			initialObjects: []runtime.Object{
				pvcDeleted(withPVCProtectionFinalizer(pvc())),
			},
			updatedPod:      withStatus(v1.PodRunning, withPVC(defaultPVCName, pod())),
			expectedActions: []clienttesting.Action{},
		},
		{
			name: "updated finished Pod -> finalizer is removed",
			initialObjects: []runtime.Object{
				pvcDeleted(withPVCProtectionFinalizer(pvc())),
			},
			updatedPod: withStatus(v1.PodSucceeded, withPVC(defaultPVCName, pod())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		{
			name: "updated unscheduled Pod -> finalizer is removed",
			initialObjects: []runtime.Object{
				pvcDeleted(withPVCProtectionFinalizer(pvc())),
			},
			updatedPod: unscheduled(withPVC(defaultPVCName, pod())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
		{
			name: "deleted running Pod -> finalizer is removed",
			initialObjects: []runtime.Object{
				pvcDeleted(withPVCProtectionFinalizer(pvc())),
			},
			deletedPod: withStatus(v1.PodRunning, withPVC(defaultPVCName, pod())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvcVer, defaultNS, pvcDeleted(pvc())),
			},
		},
	}

	for _, test := range tests {
		// Create client with initial data
		objs := test.initialObjects
		if test.updatedPV != nil {
			objs = append(objs, test.updatedPV)
		}
		if test.updatedPVC != nil {
			objs = append(objs, test.updatedPVC)
		}
		if test.updatedPod != nil {
			objs = append(objs, test.updatedPod)
		}
		client := fake.NewSimpleClientset(objs...)

		// Create informers
		// Create informers
		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		pvInformer := informers.Core().V1().PersistentVolumes()
		pvcInformer := informers.Core().V1().PersistentVolumeClaims()
		podInformer := informers.Core().V1().Pods()

		// Populate the informers with initial objects so the controller can
		// Get() and List() it.
		for _, obj := range objs {
			switch obj.(type) {
			case *v1.PersistentVolume:
				pvInformer.Informer().GetStore().Add(obj)
			case *v1.PersistentVolumeClaim:
				pvcInformer.Informer().GetStore().Add(obj)
			case *v1.Pod:
				podInformer.Informer().GetStore().Add(obj)
			default:
				t.Fatalf("Unknown initalObject type: %+v", obj)
			}
		}

		// Add reactor to inject test errors.
		for _, reactor := range test.reactors {
			client.Fake.PrependReactor(reactor.verb, reactor.resource, reactor.reactorfn)
		}

		// Create the controller
		ctrl := NewStorageObjectInUseProtectionController(pvInformer, pvcInformer, podInformer, client)

		// Start the test by simulating an event
		if test.updatedPV != nil {
			ctrl.pvAddedUpdated(test.updatedPV)
		}
		if test.updatedPVC != nil {
			ctrl.pvcAddedUpdated(test.updatedPVC)
		}
		if test.updatedPod != nil {
			ctrl.podAddedDeletedUpdated(test.updatedPod, false)
		}
		if test.deletedPod != nil {
			ctrl.podAddedDeletedUpdated(test.deletedPod, true)
		}

		// Process the controller queue until we get expected results
		timeout := time.Now().Add(10 * time.Second)
		lastReportedActionCount := 0
		for {
			if time.Now().After(timeout) {
				t.Errorf("Test %q: timed out", test.name)
				break
			}
			if ctrl.pvcQueue.Len() > 0 {
				glog.V(5).Infof("Test %q: %d events queue, processing one", test.name, ctrl.pvcQueue.Len())
				ctrl.runClaimWorker()
			}
			if ctrl.pvQueue.Len() > 0 {
				glog.V(5).Infof("Test %q: %d events queue, processing one", test.name, ctrl.pvQueue.Len())
				ctrl.runVolumeWorker()
			}

			if ctrl.pvcQueue.Len() > 0 || ctrl.pvQueue.Len() > 0 {
				// There is still some work in the queue, process it now
				continue
			}
			currentActionCount := len(client.Actions())
			if currentActionCount < len(test.expectedActions) {
				// Do not log every wait, only when the action count changes.
				if lastReportedActionCount < currentActionCount {
					glog.V(5).Infof("Test %q: got %d actions out of %d, waiting for the rest", test.name, currentActionCount, len(test.expectedActions))
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

		if !reflect.DeepEqual(actions, test.expectedActions) {
			t.Errorf("Test %q: action not expected\nExpected:\n%s\ngot:\n%s", test.name, spew.Sdump(test.expectedActions), spew.Sdump(actions))
		}

	}
}
