/*
Copyright 2024 The Kubernetes Authors.

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

package vacprotection

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

var (
	vacGVR = schema.GroupVersionResource{
		Group:    storagev1beta1.GroupName,
		Version:  "v1beta1",
		Resource: "volumeattributesclasses",
	}

	vac1                         = protectionutil.MakeVolumeAttributesClass().Name("vac1").Obj()
	vac1WithFinalizer            = protectionutil.MakeVolumeAttributesClass().Name("vac1").Finalizer(volumeutil.VACProtectionFinalizer).Obj()
	vac1TerminatingWithFinalizer = protectionutil.MakeVolumeAttributesClass().Name("vac1").Finalizer(volumeutil.VACProtectionFinalizer).Terminating().Obj()
	vac1Terminating              = protectionutil.MakeVolumeAttributesClass().Name("vac1").Terminating().Obj()

	pv1WithVAC1 = protectionutil.MakePersistentVolume().Name("pv1").VolumeAttributesClassName("vac1").Obj()
	pv1WithVAC2 = protectionutil.MakePersistentVolume().Name("pv1").VolumeAttributesClassName("vac2").Obj()
	pv2WithVAC1 = protectionutil.MakePersistentVolume().Name("pv2").VolumeAttributesClassName("vac1").Obj()

	pvc1WithVAC1            = protectionutil.MakePersistentVolumeClaim().Name("pvc1").VolumeAttributesClassName("vac1").Obj()
	pvc1WithVAC2            = protectionutil.MakePersistentVolumeClaim().Name("pvc1").VolumeAttributesClassName("vac2").Obj()
	pvc1WithVAC2CurrentVAC1 = protectionutil.MakePersistentVolumeClaim().Name("pvc1").VolumeAttributesClassName("vac2").CurrentVolumeAttributesClassName("vac1").Obj()
	pvc1WithVAC2TargetVAC1  = protectionutil.MakePersistentVolumeClaim().Name("pvc1").VolumeAttributesClassName("vac2").TargetVolumeAttributesClassName("vac1").Obj()
	pvc2WithVAC1            = protectionutil.MakePersistentVolumeClaim().Name("pvc2").VolumeAttributesClassName("vac1").Obj()
)

type reaction struct {
	verb      string
	resource  string
	reactorfn clienttesting.ReactionFunc
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

func TestVACProtectionController(t *testing.T) {
	tests := []struct {
		name string
		// Object to insert into fake kubeclient before the test starts.
		initialObjects []runtime.Object
		// Optional client reactors.
		reactors []reaction

		// VAC event to simulate. This VAC will be automatically added to
		// initialObjects.
		updatedVAC *storagev1beta1.VolumeAttributesClass

		// PV event to simulate. The updatedPV will be automatically added to
		// initialObjects.
		oldPV     *v1.PersistentVolume
		updatedPV *v1.PersistentVolume

		// PVC event to simulate. The updatedPVC will be automatically added to
		// initialObjects.
		oldPVC     *v1.PersistentVolumeClaim
		updatedPVC *v1.PersistentVolumeClaim

		// List of expected kubeclient actions that should happen during the
		// test.
		expectedActions []clienttesting.Action
	}{
		// VAC events
		//
		{
			name:       "VAC without finalizer -> finalizer is added",
			updatedVAC: vac1,
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(vacGVR, "", vac1WithFinalizer),
			},
		},
		{
			name:            "VAC with finalizer -> no action",
			updatedVAC:      vac1WithFinalizer,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:       "saving VAC finalizer fails -> controller retries",
			updatedVAC: vac1,
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "volumeattributesclasses",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// This fails
				clienttesting.NewUpdateAction(vacGVR, "", vac1WithFinalizer),
				// This fails too
				clienttesting.NewUpdateAction(vacGVR, "", vac1WithFinalizer),
				// This succeeds
				clienttesting.NewUpdateAction(vacGVR, "", vac1WithFinalizer),
			},
		},
		{
			name:       "deleted VAC with finalizer -> finalizer is removed",
			updatedVAC: vac1TerminatingWithFinalizer,
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
			},
		},
		{
			name:       "finalizer removal fails -> controller retries",
			updatedVAC: vac1TerminatingWithFinalizer,
			reactors: []reaction{
				{
					verb:      "update",
					resource:  "volumeattributesclasses",
					reactorfn: generateUpdateErrorFunc(t, 2 /* update fails twice*/),
				},
			},
			expectedActions: []clienttesting.Action{
				// Fails
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
				// Fails too
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
				// Succeeds
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
			},
		},
		{
			name:            "deleted VAC with finalizer but it's referenced by a PV -> finalizer is not removed",
			initialObjects:  []runtime.Object{pv1WithVAC1},
			updatedVAC:      vac1TerminatingWithFinalizer,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "deleted VAC with finalizer but it's referenced by a PVC -> finalizer is not removed",
			initialObjects:  []runtime.Object{pvc1WithVAC1},
			updatedVAC:      vac1TerminatingWithFinalizer,
			expectedActions: []clienttesting.Action{},
		},
		// PV events
		//
		{
			name:           "pv changes vac and deleted VAC with finalizer -> finalizer is removed",
			initialObjects: []runtime.Object{vac1TerminatingWithFinalizer},
			oldPV:          pv1WithVAC1,
			updatedPV:      pv1WithVAC2,
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
			},
		},
		{
			name:           "pv is deleted and deleted VAC with finalizer -> finalizer is removed",
			initialObjects: []runtime.Object{vac1TerminatingWithFinalizer},
			oldPV:          pv1WithVAC1,
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
			},
		},
		{
			name:            "pv is deleted but other pv still holds terminating vac -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer, pv2WithVAC1},
			oldPV:           pv1WithVAC1,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "pv is deleted but pvc still holds terminating vac -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer, pvc1WithVAC1},
			oldPV:           pv1WithVAC1,
			expectedActions: []clienttesting.Action{},
		},
		// PVC events
		//
		{
			name:           "pvc changes vac and deleted VAC with finalizer -> finalizer is removed",
			initialObjects: []runtime.Object{vac1TerminatingWithFinalizer},
			oldPVC:         pvc1WithVAC1,
			updatedPVC:     pvc1WithVAC2,
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(vacGVR, "", vac1Terminating),
			},
		},
		{
			name:            "pvc changes vac but its status still holds terminating vac  -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer},
			oldPVC:          pvc1WithVAC1,
			updatedPVC:      pvc1WithVAC2CurrentVAC1,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "pvc changes vac but its target status still holds terminating vac  -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer},
			oldPVC:          pvc1WithVAC1,
			updatedPVC:      pvc1WithVAC2TargetVAC1,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "pvc is deleted but other pvc still holds terminating vac -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer, pvc2WithVAC1},
			oldPVC:          pvc1WithVAC1,
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "pvc is deleted but pv still holds terminating vac -> finalizer is not removed",
			initialObjects:  []runtime.Object{vac1TerminatingWithFinalizer, pv1WithVAC1},
			oldPVC:          pvc1WithVAC1,
			expectedActions: []clienttesting.Action{},
		},
	}

	for _, test := range tests {
		// Create client with initial data
		objs := test.initialObjects
		if test.updatedVAC != nil {
			objs = append(objs, test.updatedVAC)
		}
		if test.updatedPV != nil {
			objs = append(objs, test.updatedPV)
		}
		if test.updatedPVC != nil {
			objs = append(objs, test.updatedPVC)
		}

		client := fake.NewSimpleClientset(objs...)

		// Create informers
		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		pvcInformer := informers.Core().V1().PersistentVolumeClaims()
		pvInformer := informers.Core().V1().PersistentVolumes()
		vacInformer := informers.Storage().V1beta1().VolumeAttributesClasses()

		// Populate the informers with initial objects so the controller can
		// Get() it.
		for _, obj := range objs {
			switch obj.(type) {
			case *v1.PersistentVolumeClaim:
				require.NoError(t, pvcInformer.Informer().GetStore().Add(obj), "failed to add object to PVC informer")
			case *v1.PersistentVolume:
				require.NoError(t, pvInformer.Informer().GetStore().Add(obj), "failed to add object to PV informer")
			case *storagev1beta1.VolumeAttributesClass:
				require.NoError(t, vacInformer.Informer().GetStore().Add(obj), "failed to add object to VAC informer")
			default:
				t.Fatalf("Unknown initialObject type: %+v", obj)
			}
		}

		// Add reactor to inject test errors.
		for _, reactor := range test.reactors {
			client.Fake.PrependReactor(reactor.verb, reactor.resource, reactor.reactorfn)
		}

		// Create the controller
		logger, _ := ktesting.NewTestContext(t)
		ctrl, err := NewVACProtectionController(logger, client, pvcInformer, pvInformer, vacInformer)
		require.NoError(t, err, "failed to create controller")

		// Start the test by simulating an event
		if test.updatedVAC != nil {
			ctrl.vacAddedUpdated(logger, test.updatedVAC)
		}
		if test.updatedPV != nil {
			ctrl.pvUpdated(logger, test.oldPV, test.updatedPV)
		} else if test.oldPV != nil {
			ctrl.pvDeleted(logger, test.oldPV)
		}
		if test.updatedPVC != nil {
			ctrl.pvcUpdated(logger, test.oldPVC, test.updatedPVC)
		} else if test.oldPVC != nil {
			ctrl.pvcDeleted(logger, test.oldPVC)
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
				logger.V(5).Info("Non-empty events queue, processing one", "test", test.name, "queueLength", ctrl.queue.Len())
				ctrl.processNextWorkItem(context.TODO())
			}
			if ctrl.queue.Len() > 0 {
				// There is still some work in the queue, process it now
				continue
			}
			currentActionCount := len(client.Actions())
			if currentActionCount < len(test.expectedActions) {
				// Do not log evey wait, only when the action count changes.
				if lastReportedActionCount < currentActionCount {
					logger.V(5).Info("Waiting for the remaining actions", "test", test.name, "currentActionCount", currentActionCount, "expectedActionCount", len(test.expectedActions))
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
			t.Errorf("Test %q: action not expected\nExpected:\n%s\ngot:\n%s", test.name, dump.Pretty(test.expectedActions), dump.Pretty(actions))
		}
	}
}
