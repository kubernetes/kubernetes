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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	appsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// Scaler provides an interface for resources that can be scaled.
type Scaler interface {
	// Scale scales the named resource after checking preconditions. It optionally
	// retries in the event of resource version mismatch (if retry is not nil),
	// and optionally waits until the status of the resource matches newSize (if wait is not nil)
	Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, wait *RetryParams) error
	// ScaleSimple does a simple one-shot attempt at scaling - not useful on its own, but
	// a necessary building block for Scale
	ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (updatedResourceVersion string, err error)
}

func ScalerFor(kind schema.GroupKind, c internalclientset.Interface) (Scaler, error) {
	switch kind {
	case api.Kind("ReplicationController"):
		return &ReplicationControllerScaler{c.Core()}, nil
	case extensions.Kind("ReplicaSet"), apps.Kind("ReplicaSet"):
		return &ReplicaSetScaler{c.Extensions()}, nil
	case batch.Kind("Job"):
		return &JobScaler{c.Batch()}, nil // Either kind of job can be scaled with Batch interface.
	case apps.Kind("StatefulSet"):
		return &StatefulSetScaler{c.Apps()}, nil
	case extensions.Kind("Deployment"), apps.Kind("Deployment"):
		return &DeploymentScaler{c.Extensions()}, nil
	}
	return nil, fmt.Errorf("no scaler has been implemented for %q", kind)
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

// ValidateStatefulSet ensures that the preconditions match. Returns nil if they are valid, an error otherwise.
func (precondition *ScalePrecondition) ValidateStatefulSet(ps *apps.StatefulSet) error {
	if precondition.Size != -1 && int(ps.Spec.Replicas) != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(int(ps.Spec.Replicas))}
	}
	if len(precondition.ResourceVersion) != 0 && ps.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, ps.ResourceVersion}
	}
	return nil
}

// ValidateReplicationController ensures that the preconditions match.  Returns nil if they are valid, an error otherwise
func (precondition *ScalePrecondition) ValidateReplicationController(controller *api.ReplicationController) error {
	if precondition.Size != -1 && int(controller.Spec.Replicas) != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(int(controller.Spec.Replicas))}
	}
	if len(precondition.ResourceVersion) != 0 && controller.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, controller.ResourceVersion}
	}
	return nil
}

type ReplicationControllerScaler struct {
	c coreclient.ReplicationControllersGetter
}

// ScaleSimple does a simple one-shot attempt at scaling. It returns the
// resourceVersion of the replication controller if the update is successful.
func (scaler *ReplicationControllerScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	controller, err := scaler.c.ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateReplicationController(controller); err != nil {
			return "", err
		}
	}
	controller.Spec.Replicas = int32(newSize)
	updatedRC, err := scaler.c.ReplicationControllers(namespace).Update(controller)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, controller.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, controller.ResourceVersion, err}
	}
	return updatedRC.ObjectMeta.ResourceVersion, nil
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
	var updatedResourceVersion string
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, &updatedResourceVersion)
	if err := wait.PollImmediate(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		checkRC := func(rc *api.ReplicationController) bool {
			if uint(rc.Spec.Replicas) != newSize {
				// the size is changed by other party. Don't need to wait for the new change to complete.
				return true
			}
			return rc.Status.ObservedGeneration >= rc.Generation && rc.Status.Replicas == rc.Spec.Replicas
		}
		// If number of replicas doesn't change, then the update may not event
		// be sent to underlying databse (we don't send no-op changes).
		// In such case, <updatedResourceVersion> will have value of the most
		// recent update (which may be far in the past) so we may get "too old
		// RV" error from watch or potentially no ReplicationController events
		// will be deliver, since it may already be in the expected state.
		// To protect from these two, we first issue Get() to ensure that we
		// are not already in the expected state.
		currentRC, err := scaler.c.ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if !checkRC(currentRC) {
			watchOptions := metav1.ListOptions{
				FieldSelector:   fields.OneTermEqualSelector("metadata.name", name).String(),
				ResourceVersion: updatedResourceVersion,
			}
			watcher, err := scaler.c.ReplicationControllers(namespace).Watch(watchOptions)
			if err != nil {
				return err
			}
			_, err = watch.Until(waitForReplicas.Timeout, watcher, func(event watch.Event) (bool, error) {
				if event.Type != watch.Added && event.Type != watch.Modified {
					return false, nil
				}
				return checkRC(event.Object.(*api.ReplicationController)), nil
			})
			if err == wait.ErrWaitTimeout {
				return fmt.Errorf("timed out waiting for %q to be synced", name)
			}
			return err
		}
	}
	return nil
}

// ValidateReplicaSet ensures that the preconditions match.  Returns nil if they are valid, an error otherwise
func (precondition *ScalePrecondition) ValidateReplicaSet(replicaSet *extensions.ReplicaSet) error {
	if precondition.Size != -1 && int(replicaSet.Spec.Replicas) != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(int(replicaSet.Spec.Replicas))}
	}
	if len(precondition.ResourceVersion) != 0 && replicaSet.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, replicaSet.ResourceVersion}
	}
	return nil
}

type ReplicaSetScaler struct {
	c extensionsclient.ReplicaSetsGetter
}

// ScaleSimple does a simple one-shot attempt at scaling. It returns the
// resourceVersion of the replicaset if the update is successful.
func (scaler *ReplicaSetScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	rs, err := scaler.c.ReplicaSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateReplicaSet(rs); err != nil {
			return "", err
		}
	}
	rs.Spec.Replicas = int32(newSize)
	updatedRS, err := scaler.c.ReplicaSets(namespace).Update(rs)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, rs.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, rs.ResourceVersion, err}
	}
	return updatedRS.ObjectMeta.ResourceVersion, nil
}

// Scale updates a ReplicaSet to a new size, with optional precondition check (if preconditions is
// not nil), optional retries (if retry is not nil), and then optionally waits for it's replica
// count to reach the new value (if wait is not nil).
func (scaler *ReplicaSetScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		rs, err := scaler.c.ReplicaSets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if rs.Initializers != nil {
			return nil
		}
		err = wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout, client.ReplicaSetHasDesiredReplicas(scaler.c, rs))

		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
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

type StatefulSetScaler struct {
	c appsclient.StatefulSetsGetter
}

// ScaleSimple does a simple one-shot attempt at scaling. It returns the
// resourceVersion of the statefulset if the update is successful.
func (scaler *StatefulSetScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	ss, err := scaler.c.StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateStatefulSet(ss); err != nil {
			return "", err
		}
	}
	ss.Spec.Replicas = int32(newSize)
	updatedStatefulSet, err := scaler.c.StatefulSets(namespace).Update(ss)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, ss.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, ss.ResourceVersion, err}
	}
	return updatedStatefulSet.ResourceVersion, nil
}

func (scaler *StatefulSetScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		job, err := scaler.c.StatefulSets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if job.Initializers != nil {
			return nil
		}
		err = wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout, client.StatefulSetHasDesiredReplicas(scaler.c, job))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
}

type JobScaler struct {
	c batchclient.JobsGetter
}

// ScaleSimple is responsible for updating job's parallelism. It returns the
// resourceVersion of the job if the update is successful.
func (scaler *JobScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
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
func (scaler *JobScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		job, err := scaler.c.Jobs(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		err = wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout, client.JobHasDesiredParallelism(scaler.c, job))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
}

// ValidateDeployment ensures that the preconditions match.  Returns nil if they are valid, an error otherwise.
func (precondition *ScalePrecondition) ValidateDeployment(deployment *extensions.Deployment) error {
	if precondition.Size != -1 && int(deployment.Spec.Replicas) != precondition.Size {
		return PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(int(deployment.Spec.Replicas))}
	}
	if len(precondition.ResourceVersion) != 0 && deployment.ResourceVersion != precondition.ResourceVersion {
		return PreconditionError{"resource version", precondition.ResourceVersion, deployment.ResourceVersion}
	}
	return nil
}

type DeploymentScaler struct {
	c extensionsclient.DeploymentsGetter
}

// ScaleSimple is responsible for updating a deployment's desired replicas
// count. It returns the resourceVersion of the deployment if the update is
// successful.
func (scaler *DeploymentScaler) ScaleSimple(namespace, name string, preconditions *ScalePrecondition, newSize uint) (string, error) {
	deployment, err := scaler.c.Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", ScaleError{ScaleGetFailure, "", err}
	}
	if preconditions != nil {
		if err := preconditions.ValidateDeployment(deployment); err != nil {
			return "", err
		}
	}

	// TODO(madhusudancs): Fix this when Scale group issues are resolved (see issue #18528).
	// For now I'm falling back to regular Deployment update operation.
	deployment.Spec.Replicas = int32(newSize)
	updatedDeployment, err := scaler.c.Deployments(namespace).Update(deployment)
	if err != nil {
		if errors.IsConflict(err) {
			return "", ScaleError{ScaleUpdateConflictFailure, deployment.ResourceVersion, err}
		}
		return "", ScaleError{ScaleUpdateFailure, deployment.ResourceVersion, err}
	}
	return updatedDeployment.ObjectMeta.ResourceVersion, nil
}

// Scale updates a deployment to a new size, with optional precondition check (if preconditions is not nil),
// optional retries (if retry is not nil), and then optionally waits for the status to reach desired count.
func (scaler *DeploymentScaler) Scale(namespace, name string, newSize uint, preconditions *ScalePrecondition, retry, waitForReplicas *RetryParams) error {
	if preconditions == nil {
		preconditions = &ScalePrecondition{-1, ""}
	}
	if retry == nil {
		// Make it try only once, immediately
		retry = &RetryParams{Interval: time.Millisecond, Timeout: time.Millisecond}
	}
	cond := ScaleCondition(scaler, preconditions, namespace, name, newSize, nil)
	if err := wait.Poll(retry.Interval, retry.Timeout, cond); err != nil {
		return err
	}
	if waitForReplicas != nil {
		deployment, err := scaler.c.Deployments(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		err = wait.Poll(waitForReplicas.Interval, waitForReplicas.Timeout, client.DeploymentHasDesiredReplicas(scaler.c, deployment))
		if err == wait.ErrWaitTimeout {
			return fmt.Errorf("timed out waiting for %q to be synced", name)
		}
		return err
	}
	return nil
}
