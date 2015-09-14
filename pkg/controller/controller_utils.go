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
	"time"

	"sync/atomic"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	CreatedByAnnotation = "kubernetes.io/created-by"

	// If a watch drops a delete event for a pod, it'll take this long
	// before a dormant controller waiting for those packets is woken up anyway. It is
	// specifically targeted at the case where some problem prevents an update
	// of expectations, without it the controller could stay asleep forever. This should
	// be set based on the expected latency of watch events.
	//
	// Currently an controller can service (create *and* observe the watch events for said
	// creation) about 10-20 pods a second, so it takes about 1 min to service
	// 500 pods. Just creation is limited to 20qps, and watching happens with ~10-30s
	// latency/pod at the scale of 3000 pods over 100 nodes.
	ExpectationsTimeout = 3 * time.Minute
)

var (
	KeyFunc = framework.DeletionHandlingMetaNamespaceKeyFunc
)

// Expectations are a way for controllers to tell the controller manager what they expect. eg:
//	ControllerExpectations: {
//		controller1: expects  2 adds in 2 minutes
//		controller2: expects  2 dels in 2 minutes
//		controller3: expects -1 adds in 2 minutes => controller3's expectations have already been met
//	}
//
// Implementation:
//	PodExpectation = pair of atomic counters to track pod creation/deletion
//	ControllerExpectationsStore = TTLStore + a PodExpectation per controller
//
// * Once set expectations can only be lowered
// * A controller isn't synced till its expectations are either fulfilled, or expire
// * Controllers that don't set expectations will get woken up for every matching pod

// ExpKeyFunc to parse out the key from a PodExpectation
var ExpKeyFunc = func(obj interface{}) (string, error) {
	if e, ok := obj.(*PodExpectations); ok {
		return e.key, nil
	}
	return "", fmt.Errorf("Could not find key for obj %#v", obj)
}

// ControllerExpectationsInterface is an interface that allows users to set and wait on expectations.
// Only abstracted out for testing.
// Warning: if using KeyFunc it is not safe to use a single ControllerExpectationsInterface with different
// types of controllers, because the keys might conflict across types.
type ControllerExpectationsInterface interface {
	GetExpectations(controllerKey string) (*PodExpectations, bool, error)
	SatisfiedExpectations(controllerKey string) bool
	DeleteExpectations(controllerKey string)
	SetExpectations(controllerKey string, add, del int) error
	ExpectCreations(controllerKey string, adds int) error
	ExpectDeletions(controllerKey string, dels int) error
	CreationObserved(controllerKey string)
	DeletionObserved(controllerKey string)
}

// ControllerExpectations is a ttl cache mapping controllers to what they expect to see before being woken up for a sync.
type ControllerExpectations struct {
	cache.Store
}

// GetExpectations returns the PodExpectations of the given controller.
func (r *ControllerExpectations) GetExpectations(controllerKey string) (*PodExpectations, bool, error) {
	if podExp, exists, err := r.GetByKey(controllerKey); err == nil && exists {
		return podExp.(*PodExpectations), true, nil
	} else {
		return nil, false, err
	}
}

// DeleteExpectations deletes the expectations of the given controller from the TTLStore.
func (r *ControllerExpectations) DeleteExpectations(controllerKey string) {
	if podExp, exists, err := r.GetByKey(controllerKey); err == nil && exists {
		if err := r.Delete(podExp); err != nil {
			glog.V(2).Infof("Error deleting expectations for controller %v: %v", controllerKey, err)
		}
	}
}

// SatisfiedExpectations returns true if the required adds/dels for the given controller have been observed.
// Add/del counts are established by the controller at sync time, and updated as pods are observed by the controller
// manager.
func (r *ControllerExpectations) SatisfiedExpectations(controllerKey string) bool {
	if podExp, exists, err := r.GetExpectations(controllerKey); exists {
		if podExp.Fulfilled() {
			return true
		} else {
			glog.V(4).Infof("Controller still waiting on expectations %#v", podExp)
			return false
		}
	} else if err != nil {
		glog.V(2).Infof("Error encountered while checking expectations %#v, forcing sync", err)
	} else {
		// When a new controller is created, it doesn't have expectations.
		// When it doesn't see expected watch events for > TTL, the expectations expire.
		//	- In this case it wakes up, creates/deletes pods, and sets expectations again.
		// When it has satisfied expectations and no pods need to be created/destroyed > TTL, the expectations expire.
		//	- In this case it continues without setting expectations till it needs to create/delete pods.
		glog.V(4).Infof("Controller %v either never recorded expectations, or the ttl expired.", controllerKey)
	}
	// Trigger a sync if we either encountered and error (which shouldn't happen since we're
	// getting from local store) or this controller hasn't established expectations.
	return true
}

// SetExpectations registers new expectations for the given controller. Forgets existing expectations.
func (r *ControllerExpectations) SetExpectations(controllerKey string, add, del int) error {
	podExp := &PodExpectations{add: int64(add), del: int64(del), key: controllerKey}
	glog.V(4).Infof("Setting expectations %+v", podExp)
	return r.Add(podExp)
}

func (r *ControllerExpectations) ExpectCreations(controllerKey string, adds int) error {
	return r.SetExpectations(controllerKey, adds, 0)
}

func (r *ControllerExpectations) ExpectDeletions(controllerKey string, dels int) error {
	return r.SetExpectations(controllerKey, 0, dels)
}

// Decrements the expectation counts of the given controller.
func (r *ControllerExpectations) lowerExpectations(controllerKey string, add, del int) {
	if podExp, exists, err := r.GetExpectations(controllerKey); err == nil && exists {
		podExp.Seen(int64(add), int64(del))
		// The expectations might've been modified since the update on the previous line.
		glog.V(4).Infof("Lowering expectations %+v", podExp)
	}
}

// CreationObserved atomically decrements the `add` expecation count of the given controller.
func (r *ControllerExpectations) CreationObserved(controllerKey string) {
	r.lowerExpectations(controllerKey, 1, 0)
}

// DeletionObserved atomically decrements the `del` expectation count of the given controller.
func (r *ControllerExpectations) DeletionObserved(controllerKey string) {
	r.lowerExpectations(controllerKey, 0, 1)
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

// GetExpectations returns the add and del expectations of the pod.
func (e *PodExpectations) GetExpectations() (int64, int64) {
	return atomic.LoadInt64(&e.add), atomic.LoadInt64(&e.del)
}

// NewControllerExpectations returns a store for PodExpectations.
func NewControllerExpectations() *ControllerExpectations {
	return &ControllerExpectations{cache.NewTTLStore(ExpKeyFunc, ExpectationsTimeout)}
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// CreateReplica creates new replicated pods according to the spec.
	CreateReplica(namespace string, controller *api.ReplicationController) error
	// CreateReplicaOnNode creates a new pod according to the spec on the specified node.
	CreateReplicaOnNode(namespace string, ds *experimental.DaemonSet, nodeName string) error
	// DeletePod deletes the pod identified by podID.
	DeletePod(namespace string, podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	KubeClient client.Interface
	Recorder   record.EventRecorder
}

func getReplicaLabelSet(template *api.PodTemplateSpec) labels.Set {
	desiredLabels := make(labels.Set)
	for k, v := range template.Labels {
		desiredLabels[k] = v
	}
	return desiredLabels
}

func getReplicaAnnotationSet(template *api.PodTemplateSpec, object runtime.Object) (labels.Set, error) {
	desiredAnnotations := make(labels.Set)
	for k, v := range template.Annotations {
		desiredAnnotations[k] = v
	}
	createdByRef, err := api.GetReference(object)
	if err != nil {
		return desiredAnnotations, fmt.Errorf("unable to get controller reference: %v", err)
	}
	createdByRefJson, err := latest.Codec.Encode(&api.SerializedReference{
		Reference: *createdByRef,
	})
	if err != nil {
		return desiredAnnotations, fmt.Errorf("unable to serialize controller reference: %v", err)
	}
	desiredAnnotations[CreatedByAnnotation] = string(createdByRefJson)
	return desiredAnnotations, nil
}

func getReplicaPrefix(controllerName string) string {
	// use the dash (if the name isn't too long) to make the pod name a bit prettier
	prefix := fmt.Sprintf("%s-", controllerName)
	if ok, _ := validation.ValidatePodName(prefix, true); !ok {
		prefix = controllerName
	}
	return prefix
}

func (r RealPodControl) CreateReplica(namespace string, controller *api.ReplicationController) error {
	desiredLabels := getReplicaLabelSet(controller.Spec.Template)
	desiredAnnotations, err := getReplicaAnnotationSet(controller.Spec.Template, controller)
	if err != nil {
		return err
	}
	prefix := getReplicaPrefix(controller.Name)

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
	if newPod, err := r.KubeClient.Pods(namespace).Create(pod); err != nil {
		r.Recorder.Eventf(controller, "FailedCreate", "Error creating: %v", err)
		return fmt.Errorf("unable to create pod replica: %v", err)
	} else {
		glog.V(4).Infof("Controller %v created pod %v", controller.Name, newPod.Name)
		r.Recorder.Eventf(controller, "SuccessfulCreate", "Created pod: %v", newPod.Name)
	}
	return nil
}

func (r RealPodControl) CreateReplicaOnNode(namespace string, ds *experimental.DaemonSet, nodeName string) error {
	desiredLabels := getReplicaLabelSet(ds.Spec.Template)
	desiredAnnotations, err := getReplicaAnnotationSet(ds.Spec.Template, ds)
	if err != nil {
		return err
	}
	prefix := getReplicaPrefix(ds.Name)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       desiredLabels,
			Annotations:  desiredAnnotations,
			GenerateName: prefix,
		},
	}
	if err := api.Scheme.Convert(&ds.Spec.Template.Spec, &pod.Spec); err != nil {
		return fmt.Errorf("unable to convert pod template: %v", err)
	}
	// if a pod does not have labels then it cannot be controlled by any controller
	if labels.Set(pod.Labels).AsSelector().Empty() {
		return fmt.Errorf("unable to create pod replica, no labels")
	}
	pod.Spec.NodeName = nodeName
	if newPod, err := r.KubeClient.Pods(namespace).Create(pod); err != nil {
		r.Recorder.Eventf(ds, "failedCreate", "Error creating: %v", err)
		return fmt.Errorf("unable to create pod replica: %v", err)
	} else {
		glog.V(4).Infof("Controller %v created pod %v", ds.Name, newPod.Name)
		r.Recorder.Eventf(ds, "successfulCreate", "Created pod: %v", newPod.Name)
	}

	return nil
}

func (r RealPodControl) DeletePod(namespace, podID string) error {
	return r.KubeClient.Pods(namespace).Delete(podID, nil)
}

// ActivePods type allows custom sorting of pods so a controller can pick the best ones to delete.
type ActivePods []*api.Pod

func (s ActivePods) Len() int      { return len(s) }
func (s ActivePods) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s ActivePods) Less(i, j int) bool {
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

// FilterActivePods returns pods that have not terminated.
func FilterActivePods(pods []api.Pod) []*api.Pod {
	var result []*api.Pod
	for i := range pods {
		if api.PodSucceeded != pods[i].Status.Phase &&
			api.PodFailed != pods[i].Status.Phase &&
			pods[i].DeletionTimestamp == nil {
			result = append(result, &pods[i])
		}
	}
	return result
}
