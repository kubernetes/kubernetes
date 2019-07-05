/*
Copyright 2019 The Kubernetes Authors.

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

package job

import (
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// GetJob uses c to get the Job in namespace ns named name. If the returned error is nil, the returned Job is valid.
func GetJob(c clientset.Interface, ns, name string) (*batchv1.Job, error) {
	return c.BatchV1().Jobs(ns).Get(name, metav1.GetOptions{})
}

// GetJobPods returns a list of Pods belonging to a Job.
func GetJobPods(c clientset.Interface, ns, jobName string) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	return c.CoreV1().Pods(ns).List(options)
}

// CreateJob uses c to create job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been created.
func CreateJob(c clientset.Interface, ns string, job *batchv1.Job) (*batchv1.Job, error) {
	return c.BatchV1().Jobs(ns).Create(job)
}

// UpdateJob uses c to updated job in namespace ns. If the returned error is nil, the returned Job is valid and has
// been updated.
func UpdateJob(c clientset.Interface, ns string, job *batchv1.Job) (*batchv1.Job, error) {
	return c.BatchV1().Jobs(ns).Update(job)
}

// UpdateJobWithRetries updates job with retries.
func UpdateJobWithRetries(c clientset.Interface, namespace, name string, applyUpdate func(*batchv1.Job)) (job *batchv1.Job, err error) {
	jobs := c.BatchV1().Jobs(namespace)
	var updateErr error
	pollErr := wait.PollImmediate(framework.Poll, JobTimeout, func() (bool, error) {
		if job, err = jobs.Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(job)
		if job, err = jobs.Update(job); err == nil {
			e2elog.Logf("Updating job %s", name)
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

// DeleteJob uses c to delete the Job named name in namespace ns. If the returned error is nil, the Job has been
// deleted.
func DeleteJob(c clientset.Interface, ns, name string) error {
	return c.BatchV1().Jobs(ns).Delete(name, nil)
}
