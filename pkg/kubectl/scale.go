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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
)

// ScalePrecondition describes a condition that must be true for the scale to take place
// If CurrentSize == -1, it is ignored.
// If CurrentResourceVersion is the empty string, it is ignored.
// Otherwise they must equal the values in the replication controller for it to be valid.
type ScalePrecondition struct {
	Size            int
	ResourceVersion string
}

// A PreconditionError is returned when a replication controller fails to match
// the scale preconditions passed to kubectl.
type PreconditionError struct {
	Precondition  string
	ExpectedValue string
	ActualValue   string
}

func (pe PreconditionError) Error() string {
	return fmt.Sprintf("Expected %s to be %s, was %s", pe.Precondition, pe.ExpectedValue, pe.ActualValue)
}

type ControllerScaleErrorType int

const (
	ControllerScaleGetFailure ControllerScaleErrorType = iota
	ControllerScaleUpdateFailure
	ControllerScaleUpdateInvalidFailure
)

// A ControllerScaleError is returned when a scale request passes
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

// ValidateReplicationController ensures that the preconditions match.  Returns nil if they are valid, an error otherwise
func (precondition *ScalePrecondition) ValidateReplicationController(controller *api.ReplicationController) error {
	if precondition.Size != -1 && controller.Spec.Replicas != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(controller.Spec.Replicas)}
	}
	if precondition.ResourceVersion != "" && controller.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, controller.ResourceVersion}
	}
	return nil
}

// ValidateJob ensures that the preconditions match.  Returns nil if they are valid, an error otherwise
func (precondition *ScalePrecondition) ValidateJob(job *experimental.Job) error {
	if precondition.Size != -1 && job.Spec.Parallelism == nil {
		return PreconditionError{"parallelism", strconv.Itoa(precondition.Size), "nil"}
	}
	if precondition.Size != -1 && *job.Spec.Parallelism != precondition.Size {
		return PreconditionError{"parallelism", strconv.Itoa(precondition.Size), strconv.Itoa(*job.Spec.Parallelism)}
	}
	if precondition.ResourceVersion != "" && job.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, job.ResourceVersion}
	}
	return nil
}

type Scaler interface {
	// Scale scales the named resource after checking preconditions. It optionally
	// retries in the event of resource version mismatch (if retry is not nil),
	// and optionally waits until the status of the resource matches newSize (if wait is not nil)
	Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, wait *RetryParams) error
	// ScaleSimple does a simple one-shot attempt at scaling - not useful on it's own, but
	// a necessary building block for Scale
	ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) error
}

func ScalerFor(kind string, c client.Interface) (Scaler, error) {
	switch kind {
	case "ReplicationController":
		return &ReplicationControllerScaler{c}, nil
	case "Job":
		return &JobScaler{c}, nil
	}
	return nil, fmt.Errorf("no scaler has been implemented for %q", kind)
}

type ReplicationControllerScaler struct {
	c client.Interface
}
type JobScaler struct {
	c client.Interface
}

// RetryParams encapsulates the retry parameters used by kubectl's scaler.
type RetryParams struct {
	Interval, Timeout time.Duration
}

func NewRetryParams(interval, timeout time.Duration) *RetryParams {
	return &RetryParams{interval, timeout}
}

// ScaleCondition is a closure around Scale that facilitates retries via util.wait
func ScaleCondition(r Scaler, precondition *ScalePrecondition, namespace, name string, count uint) wait.ConditionFunc {
	return func() (bool, error) {
		err := r.ScaleSimple(namespace, name, precondition, count)
		switch e, _ := err.(ControllerScaleError); err.(type) {
		case nil:
			return true, nil
		case ControllerScaleError:
			// if it's invalid we shouldn't keep waiting
			if e.FailureType == ControllerScaleUpdateInvalidFailure {
				return false, err
			}
			if e.FailureType == ControllerScaleUpdateFailure {
				return false, nil
			}
		}
		return false, err
	}
}

func (scaler *ReplicationControllerScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) error {
	controller, err := scaler.c.ReplicationControllers(namespace).Get(name)
	if err != nil {
		return ControllerScaleError{ControllerScaleGetFailure, "Unknown", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateReplicationController(controller); err != nil {
			return err
		}
	}
	controller.Spec.Replicas = int(newSize)
	// TODO: do retry on 409 errors here?
	if _, err := scaler.c.ReplicationControllers(namespace).Update(controller); err != nil {
		if errors.IsInvalid(err) {
			return ControllerScaleError{ControllerScaleUpdateInvalidFailure, controller.ResourceVersion, err}
		}
		return ControllerScaleError{ControllerScaleUpdateFailure, controller.ResourceVersion, err}
	}
	// TODO: do a better job of printing objects here.
	return nil
}

// Scale updates a ReplicationController to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for it's replica count to reach the new value
// (if wait is not nil).
func (scaler *ReplicationControllerScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		rc, err := scaler.c.ReplicationControllers(namespace).Get(name)
		if err != nil {
			return err
		}
		return wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout,
			client.ControllerHasDesiredReplicas(scaler.c, rc))
	}
	return nil
}

// ScaleSimple is responsible for updating job's parallelism.
func (scaler *JobScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) error {
	job, err := scaler.c.Experimental().Jobs(namespace).Get(name)
	if err != nil {
		return ControllerScaleError{ControllerScaleGetFailure, "Unknown", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateJob(job); err != nil {
			return err
		}
	}
	parallelism := int(newSize)
	job.Spec.Parallelism = &parallelism
	if _, err := scaler.c.Experimental().Jobs(namespace).Update(job); err != nil {
		if errors.IsInvalid(err) {
			return ControllerScaleError{ControllerScaleUpdateInvalidFailure, job.ResourceVersion, err}
		}
		return ControllerScaleError{ControllerScaleUpdateFailure, job.ResourceVersion, err}

	}
	return nil
}

// Scale updates a Job to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for parallelism to reach desired
// number, which can be less than requested based on job's current progress.
func (scaler *JobScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		job, err := scaler.c.Experimental().Jobs(namespace).Get(name)
		if err != nil {
			return err
		}
		return wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout,
			client.JobHasDesiredParallelism(scaler.c, job))
	}
	return nil
}
