/*
Copyright 2014 The Kubernetes Authors.

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
	"time"

	autoscalingapi "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"

	scaleclient "k8s.io/client-go/scale"
)

// Scaler provides an interface for resources that can be scaled.
type Scaler interface {
	// Scale scales the named resource after checking preconditions. It optionally
	// retries in the event of resource version mismatch (if retry is not nil),
	// and optionally waits until the status of the resource matches newSize (if wait is not nil)
	// TODO: Make the implementation of this watch-based (#56075) once #31345 is fixed.
	Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, wait *RetryParams, gr schema.GroupResource) error
	// ScaleSimple does a simple one-shot attempt at scaling - not useful on its own, but
	// a necessary building block for Scale
	ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint, gr schema.GroupResource) (updatedResourceVersion string, err error)
}

// NewScaler get a scaler for a given resource
func NewScaler(scalesGetter scaleclient.ScalesGetter) Scaler {
	return &genericScaler{scalesGetter}
}

// ScalePrecondition describes a condition that must be true for the scale to take place
// If CurrentSize == -1, it is ignored.
// If CurrentResourceVersion is the empty string, it is ignored.
// Otherwise they must equal the values in the resource for it to be valid.
type ScalePrecondition struct {
	Size            int
	ResourceVersion string
}

// A PreconditionError is returned when a resource fails to match
// the scale preconditions passed to kubectl.
type PreconditionError struct {
	Precondition  string
	ExpectedValue string
	ActualValue   string
}

func (pe PreconditionError) Error() string {
	return fmt.Sprintf("Expected %s to be %s, was %s", pe.Precondition, pe.ExpectedValue, pe.ActualValue)
}

// RetryParams encapsulates the retry parameters used by kubectl's scaler.
type RetryParams struct {
	Interval, Timeout time.Duration
}

func NewRetryParams(interval, timeout time.Duration) *RetryParams {
	return &RetryParams{interval, timeout}
}

// ScaleCondition is a closure around Scale that facilitates retries via util.wait
func ScaleCondition(r Scaler, precondition *ScalePrecondition, namespace, name string, count uint, updatedResourceVersion *string, gr schema.GroupResource) wait.ConditionFunc {
	return func() (bool, error) {
		rv, err := r.ScaleSimple(namespace, name, precondition, count, gr)
		if updatedResourceVersion != nil {
			*updatedResourceVersion = rv
		}
		// Retry only on update conflicts.
		if errors.IsConflict(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	}
}

// validateGeneric ensures that the preconditions match. Returns nil if they are valid, otherwise an error
func (precondition *ScalePrecondition) validate(scale *autoscalingapi.Scale) error {
	if precondition.Size != -1 && int(scale.Spec.Replicas) != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(int(scale.Spec.Replicas))}
	}
	if len(precondition.ResourceVersion) > 0 && scale.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, scale.ResourceVersion}
	}
	return nil
}

// genericScaler can update scales for resources in a particular namespace
type genericScaler struct {
	scaleNamespacer scaleclient.ScalesGetter
}

var _ Scaler = &genericScaler{}

// ScaleSimple updates a scale of a given resource. It returns the resourceVersion of the scale if the update was successful.
func (s *genericScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint, gr schema.GroupResource) (updatedResourceVersion string, err error) {
	scale, err := s.scaleNamespacer.Scales(namespace).Get(gr, name)
	if err != nil {
		return "", err
	}
	if preconditions != nil {
		if err := preconditions.validate(scale); err != nil {
			return "", err
		}
	}

	scale.Spec.Replicas = int32(newSize)
	updatedScale, err := s.scaleNamespacer.Scales(namespace).Update(gr, scale)
	if err != nil {
		return "", err
	}
	return updatedScale.ResourceVersion, nil
}

// Scale updates a scale of a given resource to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for the status to reach desired count.
func (s *genericScaler) Scale(namespace, resourceName string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams, gr schema.GroupResource) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(s, preconditions, namespace, resourceName, newSize, nil, gr)
	if err := wait.PollImmediate(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		return WaitForScaleHasDesiredReplicas(s.scaleNamespacer, gr, resourceName, namespace, newSize, waitForReplicas)
	}
	return nil
}

// scaleHasDesiredReplicas returns a condition that will be true if and only if the desired replica
// count for a scale (Spec) equals its updated replicas count (Status)
func scaleHasDesiredReplicas(sClient scaleclient.ScalesGetter, gr schema.GroupResource, resourceName string, namespace string, desiredReplicas int32) wait.ConditionFunc {
	return func() (bool, error) {
		actualScale, err := sClient.Scales(namespace).Get(gr, resourceName)
		if err != nil {
			return false, err
		}
		// this means the desired scale target has been reset by something else
		if actualScale.Spec.Replicas != desiredReplicas {
			return true, nil
		}
		return actualScale.Spec.Replicas == actualScale.Status.Replicas &&
			desiredReplicas == actualScale.Status.Replicas, nil
	}
}

// WaitForScaleHasDesiredReplicas waits until condition scaleHasDesiredReplicas is satisfied
// or returns error when timeout happens
func WaitForScaleHasDesiredReplicas(sClient scaleclient.ScalesGetter, gr schema.GroupResource, resourceName string, namespace string, newSize uint, waitForReplicas *RetryParams) error {
	if waitForReplicas == nil {
		return fmt.Errorf("waitForReplicas parameter cannot be nil")
	}
	err := wait.PollImmediate(
		waitForReplicas.Interval,
		waitForReplicas.Timeout,
		scaleHasDesiredReplicas(sClient, gr, resourceName, namespace, int32(newSize)))
	if err == wait.ErrWaitTimeout {
		return fmt.Errorf("timed out waiting for %q to be synced", resourceName)
	}
	return err
}
