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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/apis/batch"

	scaleclient "k8s.io/client-go/scale"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
)

// TODO: Figure out if we should be waiting on initializers in the Scale() functions below.

// Scaler provides an interface for resources that can be scaled.
type Scaler interface {
	// Scale scales the named resource after checking preconditions. It optionally
	// retries in the event of resource version mismatch (if retry is not nil),
	// and optionally waits until the status of the resource matches newSize (if wait is not nil)
	// TODO: Make the implementation of this watch-based (#56075) once #31345 is fixed.
	Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, wait *RetryParams) error
	// ScaleSimple does a simple one-shot attempt at scaling - not useful on its own, but
	// a necessary building block for Scale
	ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (updatedResourceVersion string, err error)
}

// ScalerFor gets a scaler for a given resource
func ScalerFor(kind schema.GroupKind, jobsClient batchclient.JobsGetter, scalesGetter scaleclient.ScalesGetter, gr schema.GroupResource) Scaler {
	// it seems like jobs dont't follow "normal" scale semantics.
	// For example it is not clear whether HPA could make use of it or not.
	// For more details see: https://github.com/kubernetes/kubernetes/pull/58468
	switch kind {
	case batch.Kind("Job"):
		return &jobScaler{jobsClient} // Either kind of job can be scaled with Batch interface.
	default:
		return NewScaler(scalesGetter, gr)
	}
}

// NewScaler get a scaler for a given resource
// Note that if you are trying to crate create a scaler for "job" then stop and use ScalerFor instead.
// When scaling jobs is dead, we'll remove ScalerFor method.
func NewScaler(scalesGetter scaleclient.ScalesGetter, gr schema.GroupResource) Scaler {
	return &genericScaler{scalesGetter, gr}
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

type ScaleErrorType int

const (
	ScaleGetFailure ScaleErrorType = iota
	ScaleUpdateFailure
	ScaleUpdateConflictFailure
)

// A ScaleError is returned when a scale request passes
// preconditions but fails to actually scale the controller.
type ScaleError struct {
	FailureType     ScaleErrorType
	ResourceVersion string
	ActualError     error
}

func (c ScaleError) Error() string {
	msg := fmt.Sprintf("Scaling the resource failed with: %v", c.ActualError)
	if len(c.ResourceVersion) > 0 {
		msg += fmt.Sprintf("; Current resource version %s", c.ResourceVersion)
	}
	return msg
}

// RetryParams encapsulates the retry parameters used by kubectl's scaler.
type RetryParams struct {
	Interval, Timeout time.Duration
}

func NewRetryParams(interval, timeout time.Duration) *RetryParams {
	return &RetryParams{interval, timeout}
}

// ScaleCondition is a closure around Scale that facilitates retries via util.wait
func ScaleCondition(r Scaler, precondition *ScalePrecondition, namespace, name string, count uint, updatedResourceVersion *string) wait.ConditionFunc {
	return func() (bool, error) {
		rv, err := r.ScaleSimple(namespace, name, precondition, count)
		if updatedResourceVersion != nil {
			*updatedResourceVersion = rv
		}
		switch e, _ := err.(ScaleError); err.(type) {
		case nil:
			return true, nil
		case ScaleError:
			// Retry only on update conflicts.
			if e.FailureType == ScaleUpdateConflictFailure {
				return false, nil
			}
		}
		return false, err
	}
}

// ValidateJob ensures that the preconditions match.  Returns nil if they are valid, an error otherwise.
func (precondition *ScalePrecondition) ValidateJob(job *batch.Job) error {
	if precondition.Size != -1 && job.Spec.Parallelism == nil {
		return PreconditionError{"parallelism", strconv.Itoa(precondition.Size), "nil"}
	}
	if precondition.Size != -1 && int(*job.Spec.Parallelism) != precondition.Size {
		return PreconditionError{"parallelism", strconv.Itoa(precondition.Size), strconv.Itoa(int(*job.Spec.Parallelism))}
	}
	if len(precondition.ResourceVersion) != 0 && job.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, job.ResourceVersion}
	}
	return nil
}

type jobScaler struct {
	c batchclient.JobsGetter
}

// ScaleSimple is responsible for updating job's parallelism. It returns the
// resourceVersion of the job if the update is successful.
func (scaler *jobScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	job, err := scaler.c.Jobs(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateJob(job); err != nil {
			return "", err
		}
	}
	parallelism := int32(newSize)
	job.Spec.Parallelism = &parallelism
	updatedJob, err := scaler.c.Jobs(namespace).Update(job)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, job.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, job.ResourceVersion, err}
	}
	return updatedJob.ObjectMeta.ResourceVersion, nil
}

// Scale updates a Job to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for parallelism to reach desired
// number, which can be less than requested based on job's current progress.
func (scaler *jobScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.PollImmediate(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		job, err := scaler.c.Jobs(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		err = wait.PollImmediate(waitForReplicas.Interval, waitForReplicas.Timeout, JobHasDesiredParallelism(scaler.c, job))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
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
	targetGR        schema.GroupResource
}

var _ Scaler = &genericScaler{}

// ScaleSimple updates a scale of a given resource. It returns the resourceVersion of the scale if the update was successful.
func (s *genericScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (updatedResourceVersion string, err error) {
	scale, err := s.scaleNamespacer.Scales(namespace).Get(s.targetGR, name)
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.validate(scale); err != nil {
			return "", err
		}
	}

	scale.Spec.Replicas = int32(newSize)
	updatedScale, err := s.scaleNamespacer.Scales(namespace).Update(s.targetGR, scale)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, scale.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, scale.ResourceVersion, err}
	}
	return updatedScale.ResourceVersion, nil
}

// Scale updates a scale of a given resource to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for the status to reach desired count.
func (s *genericScaler) Scale(namespace, resourceName string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(s, preconditions, namespace, resourceName, newSize, nil)
	if err := wait.PollImmediate(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		err := wait.PollImmediate(
			waitForReplicas.Interval,
			waitForReplicas.Timeout,
			scaleHasDesiredReplicas(s.scaleNamespacer, s.targetGR, resourceName, namespace, int32(newSize)))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", resourceName)
		}
		return err
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
