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

package pvprotection

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/klog"

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

const defaultPVName = "default-pv"

type reaction struct {
	verb      string
	resource  string
	reactorfn clienttesting.ReactionFunc
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

func withProtectionFinalizer(pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.Finalizers = append(pv.Finalizers, volumeutil.PVProtectionFinalizer)
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
			return true, nil, apierrors.NewForbidden(update.GetResource().GroupResource(), acc.GetName(), errors.New("Mock error"))
		}
		// Update succeeds
		return false, nil, nil
	}
}

func deleted(pv *v1.PersistentVolume) *v1.PersistentVolume {
	pv.DeletionTimestamp = &metav1.Time{}
	return pv
}

func TestPVProtectionController(t *testing.T) {
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
		// List of expected kubeclient actions that should happen during the
		// test.
		expectedActions                     []clienttesting.Action
		storageObjectInUseProtectionEnabled bool
	}{
		// PV events
		//
		{
			name:      "StorageObjectInUseProtection Enabled, PV without finalizer -> finalizer is added",
			updatedPV: pv(),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvVer, "", withProtectionFinalizer(pv())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:                                "StorageObjectInUseProtection Disabled, PV without finalizer -> finalizer is added",
			updatedPV:                           pv(),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: false,
		},
		{
			name:                                "PVC with finalizer -> no action",
			updatedPV:                           withProtectionFinalizer(pv()),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
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
				clienttesting.NewUpdateAction(pvVer, "", withProtectionFinalizer(pv())),
				// This fails too
				clienttesting.NewUpdateAction(pvVer, "", withProtectionFinalizer(pv())),
				// This succeeds
				clienttesting.NewUpdateAction(pvVer, "", withProtectionFinalizer(pv())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:      "StorageObjectInUseProtection Enabled, deleted PV with finalizer -> finalizer is removed",
			updatedPV: deleted(withProtectionFinalizer(pv())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvVer, "", deleted(pv())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:      "StorageObjectInUseProtection Disabled, deleted PV with finalizer -> finalizer is removed",
			updatedPV: deleted(withProtectionFinalizer(pv())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pvVer, "", deleted(pv())),
			},
			storageObjectInUseProtectionEnabled: false,
		},
		{
			name:      "finalizer removal fails -> controller retries",
			updatedPV: deleted(withProtectionFinalizer(pv())),
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "persistentvolumes",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// Fails
				clienttesting.NewUpdateAction(pvVer, "", deleted(pv())),
				// Fails too
				clienttesting.NewUpdateAction(pvVer, "", deleted(pv())),
				// Succeeds
				clienttesting.NewUpdateAction(pvVer, "", deleted(pv())),
			},
			storageObjectInUseProtectionEnabled: true,
		},
		{
			name:                                "deleted PVC with finalizer + PV is bound -> finalizer is not removed",
			updatedPV:                           deleted(withProtectionFinalizer(boundPV())),
			expectedActions:                     []clienttesting.Action{},
			storageObjectInUseProtectionEnabled: true,
		},
	}

	for _, test := range tests {
		// Create client with initial data
		objs := test.initialObjects
		if test.updatedPV != nil {
			objs = append(objs, test.updatedPV)
		}

		client := fake.NewSimpleClientset(objs...)

		// Create informers
		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		pvInformer := informers.Core().V1().PersistentVolumes()

		// Populate the informers with initial objects so the controller can
		// Get() it.
		for _, obj := range objs {
			switch obj.(type) {
			case *v1.PersistentVolume:
				pvInformer.Informer().GetStore().Add(obj)
			default:
				t.Fatalf("Unknown initalObject type: %+v", obj)
			}
		}

		// Add reactor to inject test errors.
		for _, reactor := range test.reactors {
			client.Fake.PrependReactor(reactor.verb, reactor.resource, reactor.reactorfn)
		}

		// Create the controller
		ctrl := NewPVProtectionController(pvInformer, client, test.storageObjectInUseProtectionEnabled)

		// Start the test by simulating an event
		if test.updatedPV != nil {
			ctrl.pvAddedUpdated(test.updatedPV)
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
				// Do not log evey wait, only when the action count changes.
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

		if !reflect.DeepEqual(actions, test.expectedActions) {
			t.Errorf("Test %q: action not expected\nExpected:\n%s\ngot:\n%s", test.name, spew.Sdump(test.expectedActions), spew.Sdump(actions))
		}

	}

}
