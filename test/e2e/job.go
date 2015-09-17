/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// How long to wait for a job to finish.
	jobTimeout = 5 * time.Minute

	// Job selector name
	jobSelector = "name"
)

var _ = Describe("Job", func() {
	f := NewFramework("job")
	parallelism := 2
	completions := 8

	It("should run a job", func() {
		By("Creating a job")
		job, err := createJob(f.Client, f.Namespace.Name, api.RestartPolicyOnFailure, 1, parallelism, completions)
		Expect(err).NotTo(HaveOccurred())
		// Cleanup jobs when we are done.
		defer func() {
			if err := f.Client.Experimental().Jobs(f.Namespace.Name).Delete(job.Name, api.NewDeleteOptions(0)); err != nil {
				Logf("Failed to cleanup job %v: %v.", job.Name, err)
			}
		}()

		By("Ensuring active pods == parallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
		Expect(err).NotTo(HaveOccurred())
		By("Ensuring job shows actibe pods")
		job, err = f.Client.Experimental().Jobs(f.Namespace.Name).Get(job.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(job.Status.Active).To(BeNumerically(">", 0))
		Expect(job.Status.Active).To(BeNumerically("<=", parallelism))

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should fail a job", func() {
		By("Creating a job")
		// negative timeout should fail the execution
		job, err := createJob(f.Client, f.Namespace.Name, api.RestartPolicyNever, -1, parallelism, completions)
		Expect(err).NotTo(HaveOccurred())
		// Cleanup jobs when we are done.
		defer func() {
			if err := f.Client.Experimental().Jobs(f.Namespace.Name).Delete(job.Name, api.NewDeleteOptions(0)); err != nil {
				Logf("Failed to cleanup job %v: %v.", job.Name, err)
			}
		}()

		By("Ensuring active pods == parallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job shows failures")
		job, err = f.Client.Experimental().Jobs(f.Namespace.Name).Get(job.Name)
		err = wait.Poll(poll, jobTimeout, func() (bool, error) {
			curr, err := f.Client.Experimental().Jobs(f.Namespace.Name).Get(job.Name)
			if err != nil {
				return false, err
			}
			return curr.Status.Unsuccessful > 0, nil
		})
	})

	It("should scale a job up", func() {
		newparallelism := 4
		By("Creating a job")
		job, err := createJob(f.Client, f.Namespace.Name, api.RestartPolicyOnFailure, 10, parallelism, completions)
		Expect(err).NotTo(HaveOccurred())
		// Cleanup jobs when we are done.
		defer func() {
			if err := f.Client.Experimental().Jobs(f.Namespace.Name).Delete(job.Name, api.NewDeleteOptions(0)); err != nil {
				Logf("Failed to cleanup job %v: %v.", job.Name, err)
			}
		}()

		By("Ensuring active pods == parallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
		Expect(err).NotTo(HaveOccurred())

		By("scale job up")
		scaler, err := kubectl.ScalerFor("Job", f.Client)
		Expect(err).NotTo(HaveOccurred())
		waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
		waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
		scaler.Scale(f.Namespace.Name, job.Name, uint(newparallelism), nil, waitForScale, waitForReplicas)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == newparallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, newparallelism)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should scale a job down", func() {
		oldparallelism := 4
		By("Creating a job")
		job, err := createJob(f.Client, f.Namespace.Name, api.RestartPolicyOnFailure, 10, oldparallelism, completions)
		Expect(err).NotTo(HaveOccurred())
		// Cleanup jobs when we are done.
		defer func() {
			if err := f.Client.Experimental().Jobs(f.Namespace.Name).Delete(job.Name, api.NewDeleteOptions(0)); err != nil {
				Logf("Failed to cleanup job %v: %v.", job.Name, err)
			}
		}()

		By("Ensuring active pods == oldparallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, oldparallelism)
		Expect(err).NotTo(HaveOccurred())

		By("scale job down")
		scaler, err := kubectl.ScalerFor("Job", f.Client)
		Expect(err).NotTo(HaveOccurred())
		waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
		waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
		err = scaler.Scale(f.Namespace.Name, job.Name, uint(parallelism), nil, waitForScale, waitForReplicas)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == parallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should stop a job", func() {
		By("Creating a job")
		job, err := createJob(f.Client, f.Namespace.Name, api.RestartPolicyOnFailure, 10, parallelism, completions)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == parallelism")
		err = waitForJobRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
		Expect(err).NotTo(HaveOccurred())

		By("scale job down")
		reaper, err := kubectl.ReaperFor("Job", f.Client)
		Expect(err).NotTo(HaveOccurred())
		timeout := 1 * time.Minute
		_, err = reaper.Stop(f.Namespace.Name, job.Name, timeout, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job was deleted")
		_, err = f.Client.Experimental().Jobs(f.Namespace.Name).Get(job.Name)
		Expect(err).To(HaveOccurred())
		Expect(errors.IsNotFound(err)).To(BeTrue())
	})
})

func createJob(c *client.Client, ns string, restartPolicy api.RestartPolicy, timeout, parallelism, completions int) (*experimental.Job, error) {
	name := "job-" + string(util.NewUUID())
	return c.Experimental().Jobs(ns).Create(&experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: experimental.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{jobSelector: name},
				},
				Spec: api.PodSpec{
					RestartPolicy: restartPolicy,
					Containers: []api.Container{
						{
							Name:    name,
							Image:   "gcr.io/google_containers/busybox",
							Command: []string{"sleep", strconv.Itoa(timeout)},
						},
					},
				},
			},
		},
	})
}

// Wait for pods to become Running.
func waitForJobRunning(c *client.Client, ns, jobName string, parallelism int) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{jobSelector: jobName}))
	return wait.Poll(poll, jobTimeout, func() (bool, error) {
		pods, err := c.Pods(ns).List(label, fields.Everything())
		if err != nil {
			return false, err
		}
		count := 0
		for _, p := range pods.Items {
			if p.Status.Phase == api.PodRunning {
				count++
			}
		}
		return count == parallelism, nil
	})
}

// Wait for job to reach completions.
func waitForJobFinish(c *client.Client, ns, jobName string, completions int) error {
	return wait.Poll(poll, jobTimeout, func() (bool, error) {
		curr, err := c.Experimental().Jobs(ns).Get(jobName)
		if err != nil {
			return false, err
		}
		return curr.Status.Successful == completions, nil
	})
}
