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

package kubectl

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// ScalePrecondition describes a condition that must be true for the scale to take place
// If CurrentSize == -1, it is ignored.
// If CurrentResourceVersion is the empty string, it is ignored.
// Otherwise they must equal the values in the replication controller for it to be valid.
type ScalePrecondition struct {
	Size            int
	ResourceVersion string
}

// PreconditionError is returned when a replication controller fails to match
// the scale preconditions passed to kubectl.
type PreconditionError struct {
	Precondition  string
	ExpectedValue string
	ActualValue   string
}

func (pe PreconditionError) Error() string {
	return fmt.Sprintf("Expected %s to be %s, was %s", pe.Precondition, pe.ExpectedValue, pe.ActualValue)
}

// ControllerScaleErrorType is an error type returned when
// scaling of a resource fails
type ControllerScaleErrorType int

const (
	// ControllerScaleGetFailure is an error returned when the scaler
	// cannot get a replication controller
	ControllerScaleGetFailure ControllerScaleErrorType = iota
	// 	ControllerScaleUpdateFailure is an error returned when the scaler
	// cannot update a replication controller
	ControllerScaleUpdateFailure
)

// ControllerScaleError is returned when a scale request passes
// preconditions but fails to actually scale the controller.
type ControllerScaleError struct {
	FailureType     ControllerScaleErrorType
	ResourceVersion string
	ActualError     error
}

func (c ControllerScaleError) Error() string {
	return fmt.Sprintf(
		"Scaling the controller failed with: %s; Current resource version %s",
		c.ActualError, c.ResourceVersion)
}

// Wait facilitates syncing primitives useful for waiting a
// replication controller to update its replicas as per its
// desired replica status
type Wait struct {
	Syncing  *sync.WaitGroup
	Replicas *sync.WaitGroup
}

// NewWait returns a new Wait
func NewWait() *Wait {
	return &Wait{
		Syncing:  &sync.WaitGroup{},
		Replicas: &sync.WaitGroup{},
	}
}

// Add delta on Wait
func (wg *Wait) Add(i int) {
	wg.Syncing.Add(i)
	wg.Replicas.Add(i)
}

// Validate ensures that the preconditions match. Returns nil if they are valid, an error otherwise
func (precondition *ScalePrecondition) Validate(controller *api.ReplicationController) error {
	if precondition.Size != -1 && controller.Spec.Replicas != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(controller.Spec.Replicas)}
	}
	if precondition.ResourceVersion != "" && controller.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, controller.ResourceVersion}
	}
	return nil
}

// Scaler is an interface implemented by those resources that support scaling such
// as replication controllers
type Scaler interface {
	// Scale scales the named resource after checking preconditions. It optionally
	// retries in the event of resource version mismatch (if retry is not nil),
	// and optionally waits until the status of the resource matches newSize (if wait is not nil)
	Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry *RetryParams, wg *Wait) error
	// ScaleSimple does a simple one-shot attempt at scaling - not useful on it's own, but
	// a necessary building block for Scale
	ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error)
}

// ScalerFor returns a scaler for the provided kind of resource if a
// scaler implementation for that kind exists
func ScalerFor(kind string, c ScalerClient) (Scaler, error) {
	switch kind {
	case "ReplicationController":
		return &ReplicationControllerScaler{c}, nil
	}
	return nil, fmt.Errorf("no scaler has been implemented for %q", kind)
}

// ReplicationControllerScaler is a wrapper around ScalerClient
// implementing replication controller specific functionality
type ReplicationControllerScaler struct {
	c ScalerClient
}

// RetryParams encapsulates the retry parameters used by kubectl's scaler.
type RetryParams struct {
	Interval, Timeout time.Duration
}

// NewRetryParams returns a new set of retry parameters
func NewRetryParams(interval, timeout time.Duration) *RetryParams {
	return &RetryParams{interval, timeout}
}

// ScaleCondition is a closure around ScaleSimple that facilitates retries via util.wait
func ScaleCondition(r Scaler, precondition *ScalePrecondition, namespace, name string, count uint) wait.ConditionFunc {
	return func() (bool, error) {
		_, err := r.ScaleSimple(namespace, name, precondition, count)
		switch e, _ := err.(ControllerScaleError); err.(type) {
		case nil:
			return true, nil
		case ControllerScaleError:
			if e.FailureType == ControllerScaleUpdateFailure {
				return false, nil
			}
		}
		return false, err
	}
}

// ScaleSimple does a simple one-shot attempt at scaling - not useful on it's own, but
// a necessary building block for Scale
func (scaler *ReplicationControllerScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	controller, err := scaler.c.GetReplicationController(namespace, name)
	if err != nil {
		return "", ControllerScaleError{ControllerScaleGetFailure, "Unknown", err}
	}
	if preconditions != nil {
		if err := preconditions.Validate(controller); err != nil {
			return "", err
		}
	}
	controller.Spec.Replicas = int(newSize)
	// TODO: do retry on 409 errors here?
	if _, err := scaler.c.UpdateReplicationController(namespace, controller); err != nil {
		return "", ControllerScaleError{ControllerScaleUpdateFailure, controller.ResourceVersion, err}
	}
	// TODO: do a better job of printing objects here.
	return "scaled", nil
}

// Scale updates a ReplicationController to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for it's replica count to reach the new value
// (if wait is not nil).
func (scaler *ReplicationControllerScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry *RetryParams, wg *Wait) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	if wg != nil {
		rc, err := scaler.c.GetReplicationController(namespace, name)
		if err != nil {
			return err
		}
		rc.Spec.Replicas = int(newSize)

		wg.Add(1)
		go scaler.c.ControllerHasDesiredReplicas(rc, Timeout, wg)
		defer wg.Replicas.Wait()
		wg.Syncing.Wait()
	}

	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	return nil
}

// ScalerClient abstracts access to ReplicationControllers.
type ScalerClient interface {
	GetReplicationController(namespace, name string) (*api.ReplicationController, error)
	UpdateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	ControllerHasDesiredReplicas(rc *api.ReplicationController, timeout time.Duration, wg *Wait) error
}

// NewScalerClient returns a new ScalerClient
func NewScalerClient(c client.Interface) ScalerClient {
	return &realScalerClient{c}
}

// realScalerClient is a ScalerClient which uses a Kube client.
type realScalerClient struct {
	client client.Interface
}

// GetReplicationController returns the replication controller with the provided namespace/name
func (c *realScalerClient) GetReplicationController(namespace, name string) (*api.ReplicationController, error) {
	return c.client.ReplicationControllers(namespace).Get(name)
}

// UpdateReplicationController updates the provided replication controller in the given namespace
func (c *realScalerClient) UpdateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
	return c.client.ReplicationControllers(namespace).Update(rc)
}

// ControllerHasDesiredReplicas is a closure around the ControllerHasDesiredReplicas function
func (c *realScalerClient) ControllerHasDesiredReplicas(rc *api.ReplicationController, timeout time.Duration, wg *Wait) error {
	return ControllerHasDesiredReplicas(c.client, rc, timeout, wg)
}

// ControllerHasDesiredReplicas accepts a replication controller and waits until it either observes that the pod
// store has the desired replicas as defined in rc.spec.replicas or until it times out
func ControllerHasDesiredReplicas(client client.Interface, rc *api.ReplicationController, timeout time.Duration, wg *Wait) error {
	notify := make(chan struct{})
	stop := make(chan struct{})
	defer func() {
		close(stop)
		wg.Replicas.Done()
	}()

	checkPods := func(obj interface{}) {
		select {
		case notify <- struct{}{}:
		case <-stop:
			return
		}
	}
	store, controller := framework.NewInformer(
		createPodLWForRC(client, rc),
		&api.Pod{},
		time.Second*30,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    checkPods,
			DeleteFunc: checkPods,
		},
	)

	go controller.Run(stop)
	// Wait until the controller has synced
	for !controller.HasSynced() {
		time.Sleep(time.Millisecond * 100)
	}
	wg.Syncing.Done()

	if rc.Status.Replicas == rc.Spec.Replicas {
		glog.V(3).Infof("rc/%s already has the desired replicas\n", rc.Name)
		return nil
	}

out:
	for {
		select {
		case <-notify:
			status := len(store.List())
			glog.V(3).Infof("Notification for rc/%s, has %d replicas, needs %d\n", rc.Name, status, rc.Spec.Replicas)
			if status == rc.Spec.Replicas {
				break out
			}
		case <-time.After(timeout):
			// Time-out
			glog.V(3).Infof("rc/%s timed out.", rc.Name)
			break out
		}
	}

	return nil
}

// createPodLWForRC returns a listwatcher for the provided replication controller
func createPodLWForRC(c client.Interface, rc *api.ReplicationController) *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return c.Pods(rc.Namespace).List(labels.Set(rc.Spec.Selector).AsSelector(), fields.Everything())
		},
		WatchFunc: func(rv string) (watch.Interface, error) {
			return c.Pods(rc.Namespace).Watch(labels.Set(rc.Spec.Selector).AsSelector(), fields.Everything(), rv)
		},
	}
}
