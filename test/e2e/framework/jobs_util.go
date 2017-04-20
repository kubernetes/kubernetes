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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
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
func NewTestJob(behavior, name string, rPol v1.RestartPolicy, parallelism, completions int32) *batch.Job {
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "Job",
		},
		Spec: batch.JobSpec{
			Parallelism:    &parallelism,
			Completions:    &completions,
			ManualSelector: newBool(false),
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
							Image:   "gcr.io/google_containers/busybox:1.24",
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
	return c.Batch().Jobs(ns).Get(name, metav1.GetOptions{})
}

// CreateJob uses c to create job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been created.
func CreateJob(c clientset.Interface, ns string, job *batch.Job) (*batch.Job, error) {
	return c.Batch().Jobs(ns).Create(job)
}

// UpdateJob uses c to updated job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been updated.
func UpdateJob(c clientset.Interface, ns string, job *batch.Job) (*batch.Job, error) {
	return c.Batch().Jobs(ns).Update(job)
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
	return c.Batch().Jobs(ns).Delete(name, nil)
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

// WaitForJobFinish uses c to wait for compeletions to complete for the Job jobName in namespace ns.
func WaitForJobFinish(c clientset.Interface, ns, jobName string, completions int32) error {
	return wait.Poll(Poll, JobTimeout, func() (bool, error) {
		curr, err := c.Batch().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.Status.Succeeded == completions, nil
	})
}

// WaitForJobFailure uses c to wait for up to timeout for the Job named jobName in namespace ns to fail.
func WaitForJobFailure(c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return wait.Poll(Poll, timeout, func() (bool, error) {
		curr, err := c.Batch().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, c := range curr.Status.Conditions {
			if c.Type == batch.JobFailed && c.Status == v1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
}

// CheckForAllJobPodsRunning uses c to check in the Job named jobName in ns is running. If the returned error is not
// nil the returned bool is true if the Job is running.
func CheckForAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) (bool, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.Core().Pods(ns).List(options)
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

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
