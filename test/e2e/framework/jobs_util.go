/*
Copyright 2017 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"time"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	jobutil "k8s.io/kubernetes/pkg/controller/job"
)

const (
	// How long to wait for a job to finish.
	JobTimeout = 15 * time.Minute

	// Job selector name
	JobSelectorKey = "job"
)

// NewTestJob returns a Job which does one of several testing behaviors. notTerminate starts a Job that will run
// effectively forever. fail starts a Job that will fail immediately. succeed starts a Job that will succeed
// immediately. randomlySucceedOrFail starts a Job that will succeed or fail randomly. failOnce fails the Job the
// first time it is run and succeeds subsequently. name is the Name of the Job. RestartPolicy indicates the restart
// policy of the containers in which the Pod is running. Parallelism is the Job's parallelism, and completions is the
// Job's required number of completions.
func NewTestJob(behavior, name string, rPol v1.RestartPolicy, parallelism, completions int32, activeDeadlineSeconds *int64, backoffLimit int32) *batch.Job {
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "Job",
		},
		Spec: batch.JobSpec{
			ActiveDeadlineSeconds: activeDeadlineSeconds,
			Parallelism:           &parallelism,
			Completions:           &completions,
			BackoffLimit:          &backoffLimit,
			ManualSelector:        newBool(false),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{JobSelectorKey: name},
				},
				Spec: v1.PodSpec{
					RestartPolicy: rPol,
					Volumes: []v1.Volume{
						{
							Name: "data",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "c",
							Image:   BusyBoxImage,
							Command: []string{},
							VolumeMounts: []v1.VolumeMount{
								{
									MountPath: "/data",
									Name:      "data",
								},
							},
						},
					},
				},
			},
		},
	}
	switch behavior {
	case "notTerminate":
		job.Spec.Template.Spec.Containers[0].Command = []string{"sleep", "1000000"}
	case "fail":
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit 1"}
	case "succeed":
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit 0"}
	case "randomlySucceedOrFail":
		// Bash's $RANDOM generates pseudorandom int in range 0 - 32767.
		// Dividing by 16384 gives roughly 50/50 chance of success.
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit $(( $RANDOM / 16384 ))"}
	case "failOnce":
		// Fail the first the container of the pod is run, and
		// succeed the second time. Checks for file on emptydir.
		// If present, succeed.  If not, create but fail.
		// Note that this cannot be used with RestartNever because
		// it always fails the first time for a pod.
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "if [[ -r /data/foo ]] ; then exit 0 ; else touch /data/foo ; exit 1 ; fi"}
	}
	return job
}

// GetJob uses c to get the Job in namespace ns named name. If the returned error is nil, the returned Job is valid.
func GetJob(c clientset.Interface, ns, name string) (*batch.Job, error) {
	return c.BatchV1().Jobs(ns).Get(name, metav1.GetOptions{})
}

// CreateJob uses c to create job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been created.
func CreateJob(c clientset.Interface, ns string, job *batch.Job) (*batch.Job, error) {
	return c.BatchV1().Jobs(ns).Create(job)
}

// UpdateJob uses c to updated job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been updated.
func UpdateJob(c clientset.Interface, ns string, job *batch.Job) (*batch.Job, error) {
	return c.BatchV1().Jobs(ns).Update(job)
}

// UpdateJobFunc updates the job object. It retries if there is a conflict, throw out error if
// there is any other errors. name is the job name, updateFn is the function updating the
// job object.
func UpdateJobFunc(c clientset.Interface, ns, name string, updateFn func(job *batch.Job)) {
	ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		job, err := GetJob(c, ns, name)
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(job)
		_, err = UpdateJob(c, ns, job)
		if err == nil {
			Logf("Successfully updated job %q", name)
			return true, nil
		}
		if errors.IsConflict(err) {
			Logf("Conflicting update to job %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update job %q: %v", name, err)
	}))
}

// DeleteJob uses c to delete the Job named name in namespace ns. If the returned error is nil, the Job has been
// deleted.
func DeleteJob(c clientset.Interface, ns, name string) error {
	return c.BatchV1().Jobs(ns).Delete(name, nil)
}

// GetJobPods returns a list of Pods belonging to a Job.
func GetJobPods(c clientset.Interface, ns, jobName string) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	return c.CoreV1().Pods(ns).List(options)
}

// WaitForAllJobPodsRunning wait for all pods for the Job named JobName in namespace ns to become Running.  Only use
// when pods will run for a long time, or it will be racy.
func WaitForAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) error {
	return wait.Poll(Poll, JobTimeout, func() (bool, error) {
		pods, err := GetJobPods(c, ns, jobName)
		if err != nil {
			return false, err
		}
		count := int32(0)
		for _, p := range pods.Items {
			if p.Status.Phase == v1.PodRunning {
				count++
			}
		}
		return count == parallelism, nil
	})
}

// WaitForJobComplete uses c to wait for compeletions to complete for the Job jobName in namespace ns.
func WaitForJobComplete(c clientset.Interface, ns, jobName string, completions int32) error {
	return wait.Poll(Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.Status.Succeeded == completions, nil
	})
}

// WaitForJobFinish uses c to wait for the Job jobName in namespace ns to finish (either Failed or Complete).
func WaitForJobFinish(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return jobutil.IsJobFinished(curr), nil
	})
}

// WaitForJobFailure uses c to wait for up to timeout for the Job named jobName in namespace ns to fail.
func WaitForJobFailure(c clientset.Interface, ns, jobName string, timeout time.Duration, reason string) error {
	return wait.Poll(Poll, timeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, c := range curr.Status.Conditions {
			if c.Type == batch.JobFailed && c.Status == v1.ConditionTrue {
				if reason == "" || reason == c.Reason {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

// WaitForJobGone uses c to wait for up to timeout for the Job named jobName in namespace ns to be removed.
func WaitForJobGone(c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return wait.Poll(Poll, timeout, func() (bool, error) {
		_, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

// CheckForAllJobPodsRunning uses c to check in the Job named jobName in ns is running. If the returned error is not
// nil the returned bool is true if the Job is running.
func CheckForAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) (bool, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(ns).List(options)
	if err != nil {
		return false, err
	}
	count := int32(0)
	for _, p := range pods.Items {
		if p.Status.Phase == v1.PodRunning {
			count++
		}
	}
	return count == parallelism, nil
}

// WaitForAllJobPodsRunning wait for all pods for the Job named jobName in namespace ns
// to be deleted.
func WaitForAllJobPodsGone(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(Poll, JobTimeout, func() (bool, error) {
		pods, err := GetJobPods(c, ns, jobName)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}

type updateJobFunc func(*batch.Job)

func UpdateJobWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateJobFunc) (job *batch.Job, err error) {
	jobs := c.BatchV1().Jobs(namespace)
	var updateErr error
	pollErr := wait.PollImmediate(Poll, JobTimeout, func() (bool, error) {
		if job, err = jobs.Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(job)
		if job, err = jobs.Update(job); err == nil {
			Logf("Updating job %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to job %q: %v", name, updateErr)
	}
	return job, pollErr
}

// WaitForJobDeleting uses c to wait for the Job jobName in namespace ns to have
// a non-nil deletionTimestamp (i.e. being deleted).
func WaitForJobDeleting(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.ObjectMeta.DeletionTimestamp != nil, nil
	})
}

func JobFinishTime(finishedJob *batch.Job) metav1.Time {
	var finishTime metav1.Time
	for _, c := range finishedJob.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return c.LastTransitionTime
		}
	}
	return finishTime
}
