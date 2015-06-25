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

package controller

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
	"sync/atomic"
)

const (
	CreatedByAnnotation = "kubernetes.io/created-by"
	updateRetries       = 1
)

// Expectations are a way for replication controllers to tell the rc manager what they expect. eg:
//	RCExpectations: {
//		rc1: expects  2 adds in 2 minutes
//		rc2: expects  2 dels in 2 minutes
//		rc3: expects -1 adds in 2 minutes => rc3's expectations have already been met
//	}
//
// Implementation:
//	PodExpectation = pair of atomic counters to track pod creation/deletion
//	RCExpectationsStore = TTLStore + a PodExpectation per rc
//
// * Once set expectations can only be lowered
// * An RC isn't synced till its expectations are either fulfilled, or expire
// * Rcs that don't set expectations will get woken up for every matching pod

// expKeyFunc to parse out the key from a PodExpectation
var expKeyFunc = func(obj interface{}) (string, error) {
	if e, ok := obj.(*PodExpectations); ok {
		return e.key, nil
	}
	return "", fmt.Errorf("Could not find key for obj %#v", obj)
}

// RCExpectationsManager is an interface that allows users to set and wait on expectations.
// Only abstracted out for testing.
type RCExpectationsManager interface {
	GetExpectations(rc *api.ReplicationController) (*PodExpectations, bool, error)
	SatisfiedExpectations(rc *api.ReplicationController) bool
	DeleteExpectations(rcKey string)
	ExpectCreations(rc *api.ReplicationController, adds int) error
	ExpectDeletions(rc *api.ReplicationController, dels int) error
	CreationObserved(rc *api.ReplicationController)
	DeletionObserved(rc *api.ReplicationController)
}

// RCExpectations is a ttl cache mapping rcs to what they expect to see before being woken up for a sync.
type RCExpectations struct {
	cache.Store
}

// GetExpectations returns the PodExpectations of the given rc.
func (r *RCExpectations) GetExpectations(rc *api.ReplicationController) (*PodExpectations, bool, error) {
	rcKey, err := rcKeyFunc(rc)
	if err != nil {
		return nil, false, err
	}
	if podExp, exists, err := r.GetByKey(rcKey); err == nil && exists {
		return podExp.(*PodExpectations), true, nil
	} else {
		return nil, false, err
	}
}

// DeleteExpectations deletes the expectations of the given RC from the TTLStore.
func (r *RCExpectations) DeleteExpectations(rcKey string) {
	if podExp, exists, err := r.GetByKey(rcKey); err == nil && exists {
		if err := r.Delete(podExp); err != nil {
			glog.V(2).Infof("Error deleting expectations for rc %v: %v", rcKey, err)
		}
	}
}

// SatisfiedExpectations returns true if the replication manager has observed the required adds/dels
// for the given rc. Add/del counts are established by the rc at sync time, and updated as pods
// are observed by the replication manager's podController.
func (r *RCExpectations) SatisfiedExpectations(rc *api.ReplicationController) bool {
	if podExp, exists, err := r.GetExpectations(rc); exists {
		if podExp.Fulfilled() {
			return true
		} else {
			glog.V(4).Infof("Controller still waiting on expectations %#v", podExp)
			return false
		}
	} else if err != nil {
		glog.V(2).Infof("Error encountered while checking expectations %#v, forcing sync", err)
	} else {
		// When a new rc is created, it doesn't have expectations.
		// When it doesn't see expected watch events for > TTL, the expectations expire.
		//	- In this case it wakes up, creates/deletes pods, and sets expectations again.
		// When it has satisfied expectations and no pods need to be created/destroyed > TTL, the expectations expire.
		//	- In this case it continues without setting expectations till it needs to create/delete pods.
		glog.V(4).Infof("Controller %v either never recorded expectations, or the ttl expired.", rc.Name)
	}
	// Trigger a sync if we either encountered and error (which shouldn't happen since we're
	// getting from local store) or this rc hasn't established expectations.
	return true
}

// setExpectations registers new expectations for the given rc. Forgets existing expectations.
func (r *RCExpectations) setExpectations(rc *api.ReplicationController, add, del int) error {
	rcKey, err := rcKeyFunc(rc)
	if err != nil {
		return err
	}
	podExp := &PodExpectations{add: int64(add), del: int64(del), key: rcKey}
	glog.V(4).Infof("Setting expectations %+v", podExp)
	return r.Add(podExp)
}

func (r *RCExpectations) ExpectCreations(rc *api.ReplicationController, adds int) error {
	return r.setExpectations(rc, adds, 0)
}

func (r *RCExpectations) ExpectDeletions(rc *api.ReplicationController, dels int) error {
	return r.setExpectations(rc, 0, dels)
}

// Decrements the expectation counts of the given rc.
func (r *RCExpectations) lowerExpectations(rc *api.ReplicationController, add, del int) {
	if podExp, exists, err := r.GetExpectations(rc); err == nil && exists {
		podExp.Seen(int64(add), int64(del))
		// The expectations might've been modified since the update on the previous line.
		glog.V(4).Infof("Lowering expectations %+v", podExp)
	}
}

// CreationObserved atomically decrements the `add` expecation count of the given replication controller.
func (r *RCExpectations) CreationObserved(rc *api.ReplicationController) {
	r.lowerExpectations(rc, 1, 0)
}

// DeletionObserved atomically decrements the `del` expectation count of the given replication controller.
func (r *RCExpectations) DeletionObserved(rc *api.ReplicationController) {
	r.lowerExpectations(rc, 0, 1)
}

// Expectations are either fulfilled, or expire naturally.
type Expectations interface {
	Fulfilled() bool
}

// PodExpectations track pod creates/deletes.
type PodExpectations struct {
	add int64
	del int64
	key string
}

// Seen decrements the add and del counters.
func (e *PodExpectations) Seen(add, del int64) {
	atomic.AddInt64(&e.add, -add)
	atomic.AddInt64(&e.del, -del)
}

// Fulfilled returns true if this expectation has been fulfilled.
func (e *PodExpectations) Fulfilled() bool {
	// TODO: think about why this line being atomic doesn't matter
	return atomic.LoadInt64(&e.add) <= 0 && atomic.LoadInt64(&e.del) <= 0
}

// getExpectations returns the add and del expectations of the pod.
func (e *PodExpectations) getExpectations() (int64, int64) {
	return atomic.LoadInt64(&e.add), atomic.LoadInt64(&e.del)
}

// NewRCExpectations returns a store for PodExpectations.
func NewRCExpectations() *RCExpectations {
	return &RCExpectations{cache.NewTTLStore(expKeyFunc, ExpectationsTimeout)}
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// createReplica creates new replicated pods according to the spec.
	createReplica(namespace string, controller *api.ReplicationController) error
	// deletePod deletes the pod identified by podID.
	deletePod(namespace string, podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	kubeClient client.Interface
	recorder   record.EventRecorder
}

func (r RealPodControl) createReplica(namespace string, controller *api.ReplicationController) error {
	desiredLabels := make(labels.Set)
	for k, v := range controller.Spec.Template.Labels {
		desiredLabels[k] = v
	}
	desiredAnnotations := make(labels.Set)
	for k, v := range controller.Spec.Template.Annotations {
		desiredAnnotations[k] = v
	}

	createdByRef, err := api.GetReference(controller)
	if err != nil {
		return fmt.Errorf("unable to get controller reference: %v", err)
	}
	createdByRefJson, err := latest.Codec.Encode(&api.SerializedReference{
		Reference: *createdByRef,
	})
	if err != nil {
		return fmt.Errorf("unable to serialize controller reference: %v", err)
	}

	desiredAnnotations[CreatedByAnnotation] = string(createdByRefJson)

	// use the dash (if the name isn't too long) to make the pod name a bit prettier
	prefix := fmt.Sprintf("%s-", controller.Name)
	if ok, _ := validation.ValidatePodName(prefix, true); !ok {
		prefix = controller.Name
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       desiredLabels,
			Annotations:  desiredAnnotations,
			GenerateName: prefix,
		},
	}
	if err := api.Scheme.Convert(&controller.Spec.Template.Spec, &pod.Spec); err != nil {
		return fmt.Errorf("unable to convert pod template: %v", err)
	}
	if labels.Set(pod.Labels).AsSelector().Empty() {
		return fmt.Errorf("unable to create pod replica, no labels")
	}
	if newPod, err := r.kubeClient.Pods(namespace).Create(pod); err != nil {
		r.recorder.Eventf(controller, "failedCreate", "Error creating: %v", err)
		return fmt.Errorf("unable to create pod replica: %v", err)
	} else {
		glog.V(4).Infof("Controller %v created pod %v", controller.Name, newPod.Name)
		r.recorder.Eventf(controller, "successfulCreate", "Created pod: %v", newPod.Name)
	}
	return nil
}

func (r RealPodControl) deletePod(namespace, podID string) error {
	return r.kubeClient.Pods(namespace).Delete(podID, nil)
}

// activePods type allows custom sorting of pods so an rc can pick the best ones to delete.
type activePods []*api.Pod

func (s activePods) Len() int      { return len(s) }
func (s activePods) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s activePods) Less(i, j int) bool {
	// Unassigned < assigned
	if s[i].Spec.NodeName == "" && s[j].Spec.NodeName != "" {
		return true
	}
	// PodPending < PodUnknown < PodRunning
	m := map[api.PodPhase]int{api.PodPending: 0, api.PodUnknown: 1, api.PodRunning: 2}
	if m[s[i].Status.Phase] != m[s[j].Status.Phase] {
		return m[s[i].Status.Phase] < m[s[j].Status.Phase]
	}
	// Not ready < ready
	if !api.IsPodReady(s[i]) && api.IsPodReady(s[j]) {
		return true
	}
	return false
}

// filterActivePods returns pods that have not terminated.
func filterActivePods(pods []api.Pod) []*api.Pod {
	var result []*api.Pod
	for i := range pods {
		if api.PodSucceeded != pods[i].Status.Phase &&
			api.PodFailed != pods[i].Status.Phase {
			result = append(result, &pods[i])
		}
	}
	return result
}

// updateReplicaCount attempts to update the Status.Replicas of the given controller, with a single GET/PUT retry.
func updateReplicaCount(rcClient client.ReplicationControllerInterface, controller api.ReplicationController, numReplicas int) (updateErr error) {
	// This is the steady state. It happens when the rc doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if controller.Status.Replicas == numReplicas &&
		controller.Generation == controller.Status.ObservedGeneration {
		return nil
	}
	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	generation := controller.Generation

	var getErr error
	for i, rc := 0, &controller; ; i++ {
		glog.V(4).Infof("Updating replica count for rc: %v, %d->%d (need %d), sequence No: %v->%v",
			controller.Name, controller.Status.Replicas, numReplicas, controller.Spec.Replicas, controller.Status.ObservedGeneration, generation)

		rc.Status = api.ReplicationControllerStatus{Replicas: numReplicas, ObservedGeneration: generation}
		_, updateErr = rcClient.Update(rc)
		if updateErr == nil || i >= updateRetries {
			return updateErr
		}
		// Update the controller with the latest resource version for the next poll
		if rc, getErr = rcClient.Get(controller.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
	// Failed 2 updates one of which was with the latest controller, return the update error
	return
}
