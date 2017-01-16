/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// How long to wait for a cronjob
	cronJobTimeout = 5 * time.Minute
)

var (
	CronJobGroupVersionResource      = schema.GroupVersionResource{Group: batch.GroupName, Version: "v2alpha1", Resource: "cronjobs"}
	ScheduledJobGroupVersionResource = schema.GroupVersionResource{Group: batch.GroupName, Version: "v2alpha1", Resource: "scheduledjobs"}
	BatchV2Alpha1GroupVersion        = schema.GroupVersion{Group: batch.GroupName, Version: "v2alpha1"}
)

var _ = framework.KubeDescribe("CronJob", func() {
	f := framework.NewDefaultGroupVersionFramework("cronjob", BatchV2Alpha1GroupVersion)

	BeforeEach(func() {
		framework.SkipIfMissingResource(f.ClientPool, CronJobGroupVersionResource, f.Namespace.Name)
	})

	// multiple jobs running at once
	It("should schedule multiple jobs concurrently", func() {
		By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batch.AllowConcurrent, true)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring more than one job is running at a time")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring at least two running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.Batch().Jobs(f.Namespace.Name).List(v1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		activeJobs := filterActiveJobs(jobs)
		Expect(len(activeJobs) >= 2).To(BeTrue())

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})

	// suspended should not schedule jobs
	It("should not schedule jobs when suspended [Slow]", func() {
		By("Creating a suspended cronjob")
		cronJob := newTestCronJob("suspended", "*/1 * * * ?", batch.AllowConcurrent, true)
		cronJob.Spec.Suspend = newBool(true)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring no jobs are scheduled")
		err = waitForNoJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, false)
		Expect(err).To(HaveOccurred())

		By("Ensuring no job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.Batch().Jobs(f.Namespace.Name).List(v1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(jobs.Items).To(HaveLen(0))

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})

	// only single active job is allowed for ForbidConcurrent
	It("should not schedule new jobs when ForbidConcurrent [Slow]", func() {
		By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batch.ForbidConcurrent, true)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(cronJob.Status.Active).Should(HaveLen(1))

		By("Ensuring exaclty one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.Batch().Jobs(f.Namespace.Name).List(v1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		activeJobs := filterActiveJobs(jobs)
		Expect(activeJobs).To(HaveLen(1))

		By("Ensuring no more jobs are scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		Expect(err).To(HaveOccurred())

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})

	// only single active job is allowed for ReplaceConcurrent
	It("should replace jobs when ReplaceConcurrent", func() {
		By("Creating a ReplaceConcurrent cronjob")
		cronJob := newTestCronJob("replace", "*/1 * * * ?", batch.ReplaceConcurrent, true)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(cronJob.Status.Active).Should(HaveLen(1))

		By("Ensuring exaclty one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.Batch().Jobs(f.Namespace.Name).List(v1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		activeJobs := filterActiveJobs(jobs)
		Expect(activeJobs).To(HaveLen(1))

		By("Ensuring the job is replaced with a new one")
		err = waitForJobReplaced(f.ClientSet, f.Namespace.Name, jobs.Items[0].Name)
		Expect(err).NotTo(HaveOccurred())

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})

	// shouldn't give us unexpected warnings
	It("should not emit unexpected warnings", func() {
		By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batch.AllowConcurrent, false)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring at least two jobs and at least one finished job exists by listing jobs explicitly")
		err = waitForJobsAtLeast(f.ClientSet, f.Namespace.Name, 2)
		Expect(err).NotTo(HaveOccurred())
		err = waitForAnyFinishedJob(f.ClientSet, f.Namespace.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring no unexpected event has happened")
		err = checkNoEventWithReason(f.ClientSet, f.Namespace.Name, cronJob.Name, []string{"MissingJob", "UnexpectedJob"})
		Expect(err).NotTo(HaveOccurred())

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})

	// deleted jobs should be removed from the active list
	It("should remove from active list jobs that have been deleted", func() {
		By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batch.ForbidConcurrent, true)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(cronJob.Status.Active).Should(HaveLen(1))

		By("Deleting the job")
		job := cronJob.Status.Active[0]
		reaper, err := kubectl.ReaperFor(batchinternal.Kind("Job"), f.InternalClientset)
		Expect(err).NotTo(HaveOccurred())
		timeout := 1 * time.Minute
		err = reaper.Stop(f.Namespace.Name, job.Name, timeout, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring job was deleted")
		_, err = getJob(f.ClientSet, f.Namespace.Name, job.Name)
		Expect(err).To(HaveOccurred())
		Expect(errors.IsNotFound(err)).To(BeTrue())

		By("Ensuring there are no active jobs in the cronjob")
		err = waitForNoJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, true)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring MissingJob event has occured")
		err = checkNoEventWithReason(f.ClientSet, f.Namespace.Name, cronJob.Name, []string{"MissingJob"})
		Expect(err).To(HaveOccurred())

		By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		Expect(err).NotTo(HaveOccurred())
	})
})

// newTestCronJob returns a cronjob which does one of several testing behaviors.
func newTestCronJob(name, schedule string, concurrencyPolicy batch.ConcurrencyPolicy, sleep bool) *batch.CronJob {
	parallelism := int32(1)
	completions := int32(1)
	sj := &batch.CronJob{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: batch.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: concurrencyPolicy,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Parallelism: &parallelism,
					Completions: &completions,
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyOnFailure,
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
									Name:  "c",
									Image: "gcr.io/google_containers/busybox:1.24",
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
			},
		},
	}
	if sleep {
		sj.Spec.JobTemplate.Spec.Template.Spec.Containers[0].Command = []string{"sleep", "300"}
	}
	return sj
}

func createCronJob(c clientset.Interface, ns string, cronJob *batch.CronJob) (*batch.CronJob, error) {
	return c.BatchV2alpha1().CronJobs(ns).Create(cronJob)
}

func getCronJob(c clientset.Interface, ns, name string) (*batch.CronJob, error) {
	return c.BatchV2alpha1().CronJobs(ns).Get(name, metav1.GetOptions{})
}

func deleteCronJob(c clientset.Interface, ns, name string) error {
	return c.BatchV2alpha1().CronJobs(ns).Delete(name, nil)
}

// Wait for at least given amount of active jobs.
func waitForActiveJobs(c clientset.Interface, ns, cronJobName string, active int) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		curr, err := c.BatchV2alpha1().CronJobs(ns).Get(cronJobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return len(curr.Status.Active) >= active, nil
	})
}

// Wait for jobs to appear in the active list of a cronjob or not.
// When failIfNonEmpty is set, this fails if the active set of jobs is still non-empty after
// the timeout. When failIfNonEmpty is not set, this fails if the active set of jobs is still
// empty after the timeout.
func waitForNoJobs(c clientset.Interface, ns, jobName string, failIfNonEmpty bool) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		curr, err := c.BatchV2alpha1().CronJobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if failIfNonEmpty {
			return len(curr.Status.Active) == 0, nil
		} else {
			return len(curr.Status.Active) != 0, nil
		}
	})
}

// Wait for a job to be replaced with a new one.
func waitForJobReplaced(c clientset.Interface, ns, previousJobName string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.Batch().Jobs(ns).List(v1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(jobs.Items) > 1 {
			return false, fmt.Errorf("More than one job is running %+v", jobs.Items)
		} else if len(jobs.Items) == 0 {
			framework.Logf("Warning: Found 0 jobs in namespace %v", ns)
			return false, nil
		}
		return jobs.Items[0].Name != previousJobName, nil
	})
}

// waitForJobsAtLeast waits for at least a number of jobs to appear.
func waitForJobsAtLeast(c clientset.Interface, ns string, atLeast int) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.Batch().Jobs(ns).List(v1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(jobs.Items) >= atLeast, nil
	})
}

// waitForAnyFinishedJob waits for any completed job to appear.
func waitForAnyFinishedJob(c clientset.Interface, ns string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.Batch().Jobs(ns).List(v1.ListOptions{})
		if err != nil {
			return false, err
		}
		for i := range jobs.Items {
			if job.IsJobFinished(&jobs.Items[i]) {
				return true, nil
			}
		}
		return false, nil
	})
}

// checkNoEventWithReason checks no events with a reason within a list has occured
func checkNoEventWithReason(c clientset.Interface, ns, cronJobName string, reasons []string) error {
	sj, err := c.BatchV2alpha1().CronJobs(ns).Get(cronJobName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("Error in getting cronjob %s/%s: %v", ns, cronJobName, err)
	}
	events, err := c.Core().Events(ns).Search(sj)
	if err != nil {
		return fmt.Errorf("Error in listing events: %s", err)
	}
	for _, e := range events.Items {
		for _, reason := range reasons {
			if e.Reason == reason {
				return fmt.Errorf("Found event with reason %s: %#v", reason, e)
			}
		}
	}
	return nil
}

func filterActiveJobs(jobs *batchv1.JobList) (active []*batchv1.Job) {
	for i := range jobs.Items {
		j := jobs.Items[i]
		if !job.IsJobFinished(&j) {
			active = append(active, &j)
		}
	}
	return
}
