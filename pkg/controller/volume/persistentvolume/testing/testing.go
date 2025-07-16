/*
Copyright 2019 The Kubernetes Authors.

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

package testing

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"

	"github.com/google/go-cmp/cmp" //nolint:depguard
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

// ErrVersionConflict is the error returned when resource version of requested
// object conflicts with the object in storage.
var ErrVersionConflict = errors.New("VersionError")

// VolumeReactor is a core.Reactor that simulates etcd and API server. It
// stores:
//   - Latest version of claims volumes saved by the controller.
//   - Queue of all saves (to simulate "volume/claim updated" events). This queue
//     contains all intermediate state of an object - e.g. a claim.VolumeName
//     is updated first and claim.Phase second. This queue will then contain both
//     updates as separate entries.
//   - Number of changes since the last call to VolumeReactor.syncAll().
//   - Optionally, volume and claim fake watchers which should be the same ones
//     used by the controller. Any time an event function like deleteVolumeEvent
//     is called to simulate an event, the reactor's stores are updated and the
//     controller is sent the event via the fake watcher.
//   - Optionally, list of error that should be returned by reactor, simulating
//     etcd / API server failures. These errors are evaluated in order and every
//     error is returned only once. I.e. when the reactor finds matching
//     ReactorError, it return appropriate error and removes the ReactorError from
//     the list.
type VolumeReactor struct {
	volumes              map[string]*v1.PersistentVolume
	claims               map[string]*v1.PersistentVolumeClaim
	changedObjects       []interface{}
	changedSinceLastSync int
	fakeVolumeWatch      *watch.FakeWatcher
	fakeClaimWatch       *watch.FakeWatcher
	lock                 sync.RWMutex
	errors               []ReactorError
	watchers             map[schema.GroupVersionResource]map[string][]*watch.RaceFreeFakeWatcher
}

// ReactorError is an error that is returned by test reactor (=simulated
// etcd+/API server) when an action performed by the reactor matches given verb
// ("get", "update", "create", "delete" or "*"") on given resource
// ("persistentvolumes", "persistentvolumeclaims" or "*").
type ReactorError struct {
	Verb     string
	Resource string
	Error    error
}

// React is a callback called by fake kubeClient from the controller.
// In other words, every claim/volume change performed by the controller ends
// here.
// This callback checks versions of the updated objects and refuse those that
// are too old (simulating real etcd).
// All updated objects are stored locally to keep track of object versions and
// to evaluate test results.
// All updated objects are also inserted into changedObjects queue and
// optionally sent back to the controller via its watchers.
func (r *VolumeReactor) React(ctx context.Context, action core.Action) (handled bool, ret runtime.Object, err error) {
	r.lock.Lock()
	defer r.lock.Unlock()
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Reactor got operation", "resource", action.GetResource(), "verb", action.GetVerb())

	// Inject error when requested
	err = r.injectReactError(ctx, action)
	if err != nil {
		return true, nil, err
	}

	// Test did not request to inject an error, continue simulating API server.
	switch {
	case action.Matches("create", "persistentvolumes"):
		obj := action.(core.UpdateAction).GetObject()
		volume := obj.(*v1.PersistentVolume)

		// check the volume does not exist
		_, found := r.volumes[volume.Name]
		if found {
			return true, nil, fmt.Errorf("cannot create volume %s: volume already exists", volume.Name)
		}

		// mimic apiserver defaulting
		if volume.Spec.VolumeMode == nil {
			volume.Spec.VolumeMode = new(v1.PersistentVolumeMode)
			*volume.Spec.VolumeMode = v1.PersistentVolumeFilesystem
		}

		// Store the updated object to appropriate places.
		r.volumes[volume.Name] = volume
		for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
			w.Add(volume)
		}
		r.changedObjects = append(r.changedObjects, volume)
		r.changedSinceLastSync++
		logger.V(4).Info("Created volume", "volumeName", volume.Name)
		return true, volume, nil

	case action.Matches("create", "persistentvolumeclaims"):
		obj := action.(core.UpdateAction).GetObject()
		claim := obj.(*v1.PersistentVolumeClaim)

		// check the claim does not exist
		_, found := r.claims[claim.Name]
		if found {
			return true, nil, fmt.Errorf("cannot create claim %s: claim already exists", claim.Name)
		}

		// Store the updated object to appropriate places.
		r.claims[claim.Name] = claim
		for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
			w.Add(claim)
		}
		r.changedObjects = append(r.changedObjects, claim)
		r.changedSinceLastSync++
		logger.V(4).Info("Created claim", "PVC", klog.KObj(claim))
		return true, claim, nil

	case action.Matches("update", "persistentvolumes"):
		obj := action.(core.UpdateAction).GetObject()
		volume := obj.(*v1.PersistentVolume)

		// Check and bump object version
		storedVolume, found := r.volumes[volume.Name]
		if found {
			storedVer, _ := strconv.Atoi(storedVolume.ResourceVersion)
			requestedVer, _ := strconv.Atoi(volume.ResourceVersion)
			if storedVer != requestedVer {
				return true, obj, ErrVersionConflict
			}
			if reflect.DeepEqual(storedVolume, volume) {
				logger.V(4).Info("Nothing updated volume", "volumeName", volume.Name)
				return true, volume, nil
			}
			// Don't modify the existing object
			volume = volume.DeepCopy()
			volume.ResourceVersion = strconv.Itoa(storedVer + 1)
		} else {
			return true, nil, fmt.Errorf("cannot update volume %s: volume not found", volume.Name)
		}

		// Store the updated object to appropriate places.
		for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
			w.Modify(volume)
		}
		r.volumes[volume.Name] = volume
		r.changedObjects = append(r.changedObjects, volume)
		r.changedSinceLastSync++
		logger.V(4).Info("Saved updated volume", "volumeName", volume.Name)
		return true, volume, nil

	case action.Matches("update", "persistentvolumeclaims"):
		obj := action.(core.UpdateAction).GetObject()
		claim := obj.(*v1.PersistentVolumeClaim)

		// Check and bump object version
		storedClaim, found := r.claims[claim.Name]
		if found {
			storedVer, _ := strconv.Atoi(storedClaim.ResourceVersion)
			requestedVer, _ := strconv.Atoi(claim.ResourceVersion)
			if storedVer != requestedVer {
				return true, obj, ErrVersionConflict
			}
			if reflect.DeepEqual(storedClaim, claim) {
				logger.V(4).Info("Nothing updated claim", "PVC", klog.KObj(claim))
				return true, claim, nil
			}
			// Don't modify the existing object
			claim = claim.DeepCopy()
			claim.ResourceVersion = strconv.Itoa(storedVer + 1)
		} else {
			return true, nil, fmt.Errorf("cannot update claim %s: claim not found", claim.Name)
		}

		// Store the updated object to appropriate places.
		for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
			w.Modify(claim)
		}
		r.claims[claim.Name] = claim
		r.changedObjects = append(r.changedObjects, claim)
		r.changedSinceLastSync++
		logger.V(4).Info("Saved updated claim", "PVC", klog.KObj(claim))
		return true, claim, nil

	case action.Matches("get", "persistentvolumes"):
		name := action.(core.GetAction).GetName()
		volume, found := r.volumes[name]
		if found {
			logger.V(4).Info("GetVolume: found volume", "volumeName", volume.Name)
			return true, volume.DeepCopy(), nil
		}
		logger.V(4).Info("GetVolume: volume not found", "volumeName", name)
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), name)

	case action.Matches("get", "persistentvolumeclaims"):
		name := action.(core.GetAction).GetName()
		nameSpace := action.(core.GetAction).GetNamespace()
		claim, found := r.claims[name]
		if found {
			logger.V(4).Info("GetClaim: found claim", "PVC", klog.KObj(claim))
			return true, claim.DeepCopy(), nil
		}
		logger.V(4).Info("GetClaim: claim not found", "PVC", klog.KRef(nameSpace, name))
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), name)

	case action.Matches("delete", "persistentvolumes"):
		name := action.(core.DeleteAction).GetName()
		logger.V(4).Info("Deleted volume", "volumeName", name)
		obj, found := r.volumes[name]
		if found {
			delete(r.volumes, name)
			for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
				w.Delete(obj)
			}
			r.changedSinceLastSync++
			return true, nil, nil
		}
		return true, nil, fmt.Errorf("cannot delete volume %s: not found", name)

	case action.Matches("delete", "persistentvolumeclaims"):
		name := action.(core.DeleteAction).GetName()
		nameSpace := action.(core.DeleteAction).GetNamespace()
		logger.V(4).Info("Deleted claim", "PVC", klog.KRef(nameSpace, name))
		obj, found := r.claims[name]
		if found {
			delete(r.claims, name)
			for _, w := range r.getWatches(action.GetResource(), action.GetNamespace()) {
				w.Delete(obj)
			}
			r.changedSinceLastSync++
			return true, nil, nil
		}
		return true, nil, fmt.Errorf("cannot delete claim %s: not found", name)
	}

	return false, nil, nil
}

// Watch watches objects from the VolumeReactor. Watch returns a channel which
// will push added / modified / deleted object.
func (r *VolumeReactor) Watch(gvr schema.GroupVersionResource, ns string) (watch.Interface, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	fakewatcher := watch.NewRaceFreeFake()

	if _, exists := r.watchers[gvr]; !exists {
		r.watchers[gvr] = make(map[string][]*watch.RaceFreeFakeWatcher)
	}
	r.watchers[gvr][ns] = append(r.watchers[gvr][ns], fakewatcher)
	return fakewatcher, nil
}

func (r *VolumeReactor) getWatches(gvr schema.GroupVersionResource, ns string) []*watch.RaceFreeFakeWatcher {
	watches := []*watch.RaceFreeFakeWatcher{}
	if r.watchers[gvr] != nil {
		if w := r.watchers[gvr][ns]; w != nil {
			watches = append(watches, w...)
		}
		if ns != metav1.NamespaceAll {
			if w := r.watchers[gvr][metav1.NamespaceAll]; w != nil {
				watches = append(watches, w...)
			}
		}
	}
	return watches
}

// injectReactError returns an error when the test requested given action to
// fail. nil is returned otherwise.
func (r *VolumeReactor) injectReactError(ctx context.Context, action core.Action) error {
	if len(r.errors) == 0 {
		// No more errors to inject, everything should succeed.
		return nil
	}
	logger := klog.FromContext(ctx)
	for i, expected := range r.errors {
		logger.V(4).Info("Trying to match resource verb", "resource", action.GetResource(), "verb", action.GetVerb(), "expectedResource", expected.Resource, "expectedVerb", expected.Verb)
		if action.Matches(expected.Verb, expected.Resource) {
			// That's the action we're waiting for, remove it from injectedErrors
			r.errors = append(r.errors[:i], r.errors[i+1:]...)
			logger.V(4).Info("Reactor found matching error", "index", i, "expectedResource", expected.Resource, "expectedVerb", expected.Verb, "err", expected.Error)
			return expected.Error
		}
	}
	return nil
}

// CheckVolumes compares all expectedVolumes with set of volumes at the end of
// the test and reports differences.
func (r *VolumeReactor) CheckVolumes(expectedVolumes []*v1.PersistentVolume) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	expectedMap := make(map[string]*v1.PersistentVolume)
	gotMap := make(map[string]*v1.PersistentVolume)
	// Clear any ResourceVersion from both sets
	for _, v := range expectedVolumes {
		// Don't modify the existing object
		v := v.DeepCopy()
		v.ResourceVersion = ""
		if v.Spec.ClaimRef != nil {
			v.Spec.ClaimRef.ResourceVersion = ""
		}
		expectedMap[v.Name] = v
	}
	for _, v := range r.volumes {
		// We must clone the volume because of golang race check - it was
		// written by the controller without any locks on it.
		v := v.DeepCopy()
		v.ResourceVersion = ""
		if v.Spec.ClaimRef != nil {
			v.Spec.ClaimRef.ResourceVersion = ""
		}
		gotMap[v.Name] = v
	}
	if !reflect.DeepEqual(expectedMap, gotMap) {
		// Print ugly but useful diff of expected and received objects for
		// easier debugging.
		return fmt.Errorf("Volume check failed [A-expected, B-got]: %s", cmp.Diff(expectedMap, gotMap))
	}
	return nil
}

// CheckClaims compares all expectedClaims with set of claims at the end of the
// test and reports differences.
func (r *VolumeReactor) CheckClaims(expectedClaims []*v1.PersistentVolumeClaim) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	expectedMap := make(map[string]*v1.PersistentVolumeClaim)
	gotMap := make(map[string]*v1.PersistentVolumeClaim)
	for _, c := range expectedClaims {
		// Don't modify the existing object
		c = c.DeepCopy()
		c.ResourceVersion = ""
		expectedMap[c.Name] = c
	}
	for _, c := range r.claims {
		// We must clone the claim because of golang race check - it was
		// written by the controller without any locks on it.
		c = c.DeepCopy()
		c.ResourceVersion = ""
		gotMap[c.Name] = c
	}
	if !reflect.DeepEqual(expectedMap, gotMap) {
		// Print ugly but useful diff of expected and received objects for
		// easier debugging.
		return fmt.Errorf("Claim check failed [A-expected, B-got result]: %s", cmp.Diff(expectedMap, gotMap))
	}
	return nil
}

// PopChange returns one recorded updated object, either *v1.PersistentVolume
// or *v1.PersistentVolumeClaim. Returns nil when there are no changes.
func (r *VolumeReactor) PopChange(ctx context.Context) interface{} {
	r.lock.Lock()
	defer r.lock.Unlock()

	if len(r.changedObjects) == 0 {
		return nil
	}

	// For debugging purposes, print the queue
	logger := klog.FromContext(ctx)
	for _, obj := range r.changedObjects {
		switch obj.(type) {
		case *v1.PersistentVolume:
			vol, _ := obj.(*v1.PersistentVolume)
			logger.V(4).Info("Reactor queue", "volumeName", vol.Name)
		case *v1.PersistentVolumeClaim:
			claim, _ := obj.(*v1.PersistentVolumeClaim)
			logger.V(4).Info("Reactor queue", "PVC", klog.KObj(claim))
		}
	}

	// Pop the first item from the queue and return it
	obj := r.changedObjects[0]
	r.changedObjects = r.changedObjects[1:]
	return obj
}

// SyncAll simulates the controller periodic sync of volumes and claim. It
// simply adds all these objects to the internal queue of updates. This method
// should be used when the test manually calls syncClaim/syncVolume. Test that
// use real controller loop (ctrl.Run()) will get periodic sync automatically.
func (r *VolumeReactor) SyncAll() {
	r.lock.Lock()
	defer r.lock.Unlock()

	for _, c := range r.claims {
		r.changedObjects = append(r.changedObjects, c)
	}
	for _, v := range r.volumes {
		r.changedObjects = append(r.changedObjects, v)
	}
	r.changedSinceLastSync = 0
}

// GetChangeCount returns changes since last sync.
func (r *VolumeReactor) GetChangeCount() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return r.changedSinceLastSync
}

// DeleteVolumeEvent simulates that a volume has been deleted in etcd and
// the controller receives 'volume deleted' event.
func (r *VolumeReactor) DeleteVolumeEvent(volume *v1.PersistentVolume) {
	r.lock.Lock()
	defer r.lock.Unlock()

	// Remove the volume from list of resulting volumes.
	delete(r.volumes, volume.Name)

	// Generate deletion event. Cloned volume is needed to prevent races (and we
	// would get a clone from etcd too).
	if r.fakeVolumeWatch != nil {
		r.fakeVolumeWatch.Delete(volume.DeepCopy())
	}
}

// DeleteClaimEvent simulates that a claim has been deleted in etcd and the
// controller receives 'claim deleted' event.
func (r *VolumeReactor) DeleteClaimEvent(claim *v1.PersistentVolumeClaim) {
	r.lock.Lock()
	defer r.lock.Unlock()

	// Remove the claim from list of resulting claims.
	delete(r.claims, claim.Name)

	// Generate deletion event. Cloned volume is needed to prevent races (and we
	// would get a clone from etcd too).
	if r.fakeClaimWatch != nil {
		r.fakeClaimWatch.Delete(claim.DeepCopy())
	}
}

// AddClaimEvent simulates that a claim has been deleted in etcd and the
// controller receives 'claim added' event.
func (r *VolumeReactor) AddClaimEvent(claim *v1.PersistentVolumeClaim) {
	r.lock.Lock()
	defer r.lock.Unlock()

	r.claims[claim.Name] = claim
	// Generate event. No cloning is needed, this claim is not stored in the
	// controller cache yet.
	if r.fakeClaimWatch != nil {
		r.fakeClaimWatch.Add(claim)
	}
}

// AddClaims adds PVCs into VolumeReactor.
func (r *VolumeReactor) AddClaims(claims []*v1.PersistentVolumeClaim) {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, claim := range claims {
		r.claims[claim.Name] = claim
	}
}

// AddVolumes adds PVs into VolumeReactor.
func (r *VolumeReactor) AddVolumes(volumes []*v1.PersistentVolume) {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, volume := range volumes {
		r.volumes[volume.Name] = volume
	}
}

// AddClaim adds a PVC into VolumeReactor.
func (r *VolumeReactor) AddClaim(claim *v1.PersistentVolumeClaim) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.claims[claim.Name] = claim
}

// AddVolume adds a PV into VolumeReactor.
func (r *VolumeReactor) AddVolume(volume *v1.PersistentVolume) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.volumes[volume.Name] = volume
}

// DeleteVolume deletes a PV by name.
func (r *VolumeReactor) DeleteVolume(name string) {
	r.lock.Lock()
	defer r.lock.Unlock()
	delete(r.volumes, name)
}

// AddClaimBoundToVolume adds a PVC and binds it to corresponding PV.
func (r *VolumeReactor) AddClaimBoundToVolume(claim *v1.PersistentVolumeClaim) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.claims[claim.Name] = claim
	if volume, ok := r.volumes[claim.Spec.VolumeName]; ok {
		volume.Status.Phase = v1.VolumeBound
	}
}

// MarkVolumeAvailable marks a PV available by name.
func (r *VolumeReactor) MarkVolumeAvailable(name string) {
	r.lock.Lock()
	defer r.lock.Unlock()
	if volume, ok := r.volumes[name]; ok {
		volume.Spec.ClaimRef = nil
		volume.Status.Phase = v1.VolumeAvailable
		volume.Annotations = nil
	}
}

// NewVolumeReactor creates a volume reactor.
func NewVolumeReactor(ctx context.Context, client *fake.Clientset, fakeVolumeWatch, fakeClaimWatch *watch.FakeWatcher, errors []ReactorError) *VolumeReactor {
	reactor := &VolumeReactor{
		volumes:         make(map[string]*v1.PersistentVolume),
		claims:          make(map[string]*v1.PersistentVolumeClaim),
		fakeVolumeWatch: fakeVolumeWatch,
		fakeClaimWatch:  fakeClaimWatch,
		errors:          errors,
		watchers:        make(map[schema.GroupVersionResource]map[string][]*watch.RaceFreeFakeWatcher),
	}
	client.AddReactor("create", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})

	client.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("update", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("update", "persistentvolumeclaims", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("get", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("delete", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	client.AddReactor("delete", "persistentvolumeclaims", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return reactor.React(ctx, action)
	})
	return reactor
}
