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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// How long to wait for a job to finish.
	jobTimeout = 15 * time.Minute

	// Job selector name
	jobSelectorKey = "job"
)

// TODO: Activate these tests for GKE when we support experimental APIs there.
var _ = Describe("Job", func() {
	f := NewFramework("job")
	parallelism := 2
	completions := 4
	lotsOfFailures := 5 // more than completions

	// Simplest case: all pods succeed promptly
	It("should run a job to completion when tasks succeed", func() {
		SkipIfProviderIs("gke")
		By("Creating a job")
		job := newTestJob("succeed", "all-succeed", api.RestartPolicyNever, parallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	// Pods sometimes fail, but eventually succeed.
	It("should run a job to completion when tasks sometimes fail and are locally restarted", func() {
		SkipIfProviderIs("gke")
		By("Creating a job")
		// 50% chance of container success, local restarts.
		job := newTestJob("randomlySucceedOrFail", "rand-local", api.RestartPolicyOnFailure, parallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	// Pods sometimes fail, but eventually succeed, after pod restarts
	It("should run a job to completion when tasks sometimes fail and are not locally restarted", func() {
		SkipIfProviderIs("gke")
		By("Creating a job")
		// 50% chance of container success, local restarts.
		job := newTestJob("randomlySucceedOrFail", "rand-non-local", api.RestartPolicyNever, parallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job reaches completions")
		err = waitForJobFinish(f.Client, f.Namespace.Name, job.Name, completions)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should keep restarting failed pods", func() {
		SkipIfProviderIs("gke")
		By("Creating a job")
		job := newTestJob("fail", "all-fail", api.RestartPolicyNever, parallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job shows many failures")
		err = wait.Poll(poll, jobTimeout, func() (bool, error) {
			curr, err := f.Client.Experimental().Jobs(f.Namespace.Name).Get(job.Name)
			if err != nil {
				return false, err
			}
			return curr.Status.Failed > lotsOfFailures, nil
		})
	})

	It("should scale a job up", func() {
		SkipIfProviderIs("gke")
		startParallelism := 1
		endParallelism := 2
		By("Creating a job")
		job := newTestJob("notTerminate", "scale-up", api.RestartPolicyNever, startParallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == startParallelism")
		err = waitForAllPodsRunning(f.Client, f.Namespace.Name, job.Name, startParallelism)
		Expect(err).NotTo(HaveOccurred())

		By("scale job up")
		scaler, err := kubectl.ScalerFor("Job", f.Client)
		Expect(err).NotTo(HaveOccurred())
		waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
		waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
		scaler.Scale(f.Namespace.Name, job.Name, uint(endParallelism), nil, waitForScale, waitForReplicas)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == endParallelism")
		err = waitForAllPodsRunning(f.Client, f.Namespace.Name, job.Name, endParallelism)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should scale a job down", func() {
		SkipIfProviderIs("gke")
		startParallelism := 2
		endParallelism := 1
		By("Creating a job")
		job := newTestJob("notTerminate", "scale-down", api.RestartPolicyNever, startParallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == startParallelism")
		err = waitForAllPodsRunning(f.Client, f.Namespace.Name, job.Name, startParallelism)
		Expect(err).NotTo(HaveOccurred())

		By("scale job down")
		scaler, err := kubectl.ScalerFor("Job", f.Client)
		Expect(err).NotTo(HaveOccurred())
		waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
		waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
		err = scaler.Scale(f.Namespace.Name, job.Name, uint(endParallelism), nil, waitForScale, waitForReplicas)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == endParallelism")
		err = waitForAllPodsRunning(f.Client, f.Namespace.Name, job.Name, endParallelism)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should stop a job", func() {
		SkipIfProviderIs("gke")
		By("Creating a job")
		job := newTestJob("notTerminate", "foo", api.RestartPolicyNever, parallelism, completions)
		job, err := createJob(f.Client, f.Namespace.Name, job)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring active pods == parallelism")
		err = waitForAllPodsRunning(f.Client, f.Namespace.Name, job.Name, parallelism)
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

// newTestJob returns a job which does one of several testing behaviors.
func newTestJob(behavior, name string, rPol api.RestartPolicy, parallelism, completions int) *experimental.Job {
	job := &experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: experimental.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{jobSelectorKey: name},
				},
				Spec: api.PodSpec{
					RestartPolicy: rPol,
					Containers: []api.Container{
						{
							Name:    "c",
							Image:   "gcr.io/google_containers/busybox",
							Command: []string{},
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
		// Dividing by 16384 gives roughly 50/50 chance of succeess.
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit $(( $RANDOM / 16384 ))"}
	}
	return job
}

func createJob(c *client.Client, ns string, job *experimental.Job) (*experimental.Job, error) {
	return c.Experimental().Jobs(ns).Create(job)
}

func deleteJob(c *client.Client, ns, name string) error {
	return c.Experimental().Jobs(ns).Delete(name, api.NewDeleteOptions(0))
}

// Wait for all pods to become Running.  Only use when pods will run for a long time, or it will be racy.
func waitForAllPodsRunning(c *client.Client, ns, jobName string, parallelism int) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{jobSelectorKey: jobName}))
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
		return curr.Status.Succeeded == completions, nil
	})
}
