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

package scale

import (
	"fmt"
	"strconv"
	"time"

	batch "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	batchclient "k8s.io/client-go/kubernetes/typed/batch/v1"
)

// ScalePrecondition is a deprecated precondition
type ScalePrecondition struct {
	Size            int
	ResourceVersion string
}

// RetryParams is a deprecated retry struct
type RetryParams struct {
	Interval, Timeout time.Duration
}

// PreconditionError is a deprecated error
type PreconditionError struct {
	Precondition  string
	ExpectedValue string
	ActualValue   string
}

func (pe PreconditionError) Error() string {
	return fmt.Sprintf("Expected %s to be %s, was %s", pe.Precondition, pe.ExpectedValue, pe.ActualValue)
}

// ScaleCondition is a closure around Scale that facilitates retries via util.wait
func scaleCondition(r *JobPsuedoScaler, precondition *ScalePrecondition, namespace, name string, count uint, updatedResourceVersion *string) wait.ConditionFunc {
	return func() (bool, error) {
		rv, err := r.ScaleSimple(namespace, name, precondition, count)
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

// JobPsuedoScaler is a deprecated scale-similar thing that doesn't obey scale semantics
type JobPsuedoScaler struct {
	JobsClient batchclient.JobsGetter
}

// ScaleSimple is responsible for updating job's parallelism. It returns the
// resourceVersion of the job if the update is successful.
func (scaler *JobPsuedoScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	job, err := scaler.JobsClient.Jobs(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	if preconditions != nil {
		if err := validateJob(job, preconditions); err != nil {
			return "", err
		}
	}
	parallelism := int32(newSize)
	job.Spec.Parallelism = &parallelism
	updatedJob, err := scaler.JobsClient.Jobs(namespace).Update(job)
	if err != nil {
		return "", err
	}
	return updatedJob.ObjectMeta.ResourceVersion, nil
}

// Scale updates a Job to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for parallelism to reach desired
// number, which can be less than requested based on job's current progress.
func (scaler *JobPsuedoScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := scaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.PollImmediate(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		job, err := scaler.JobsClient.Jobs(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		err = wait.PollImmediate(waitForReplicas.Interval, waitForReplicas.Timeout, jobHasDesiredParallelism(scaler.JobsClient, job))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
}

// JobHasDesiredParallelism returns a condition that will be true if the desired parallelism count
// for a job equals the current active counts or is less by an appropriate successful/unsuccessful count.
func jobHasDesiredParallelism(jobClient batchclient.JobsGetter, job *batch.Job) wait.ConditionFunc {
	return func() (bool, error) {
		job, err := jobClient.Jobs(job.Namespace).Get(job.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// desired parallelism can be either the exact number, in which case return immediately
		if job.Status.Active == *job.Spec.Parallelism {
			return true, nil
		}
		if job.Spec.Completions == nil {
			// A job without specified completions needs to wait for Active to reach Parallelism.
			return false, nil
		}

		// otherwise count successful
		progress := *job.Spec.Completions - job.Status.Active - job.Status.Succeeded
		return progress <= 0, nil
	}
}

func validateJob(job *batch.Job, precondition *ScalePrecondition) error {
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
