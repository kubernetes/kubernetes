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

package apps

import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/test/e2e/framework"
	jobutil "k8s.io/kubernetes/test/e2e/framework/job"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// How long to wait for a cronjob
	cronJobTimeout = 5 * time.Minute
)

var _ = SIGDescribe("CronJob", func() {
	f := framework.NewDefaultFramework("cronjob")

	sleepCommand := []string{"sleep", "300"}

	// Pod will complete instantly
	successCommand := []string{"/bin/true"}
	failureCommand := []string{"/bin/false"}

	ginkgo.BeforeEach(func() {
		framework.SkipIfMissingResource(f.DynamicClient, CronJobGroupVersionResourceBeta, f.Namespace.Name)
	})

	// multiple jobs running at once
	ginkgo.It("should schedule multiple jobs concurrently", func() {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1beta1.AllowConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring more than one job is running at a time")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		framework.ExpectNoError(err, "Failed to wait for active jobs in CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring at least two running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(len(activeJobs)).To(gomega.BeNumerically(">=", 2))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// suspended should not schedule jobs
	ginkgo.It("should not schedule jobs when suspended [Slow]", func() {
		ginkgo.By("Creating a suspended cronjob")
		cronJob := newTestCronJob("suspended", "*/1 * * * ?", batchv1beta1.AllowConcurrent,
			sleepCommand, nil, nil)
		t := true
		cronJob.Spec.Suspend = &t
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring no jobs are scheduled")
		err = waitForNoJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, false)
		framework.ExpectError(err)

		ginkgo.By("Ensuring no job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		gomega.Expect(jobs.Items).To(gomega.HaveLen(0))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// only single active job is allowed for ForbidConcurrent
	ginkgo.It("should not schedule new jobs when ForbidConcurrent [Slow]", func() {
		ginkgo.By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1beta1.ForbidConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to schedule CronJob %s", cronJob.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to get CronJob %s", cronJob.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Ensuring exactly one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(activeJobs).To(gomega.HaveLen(1))

		ginkgo.By("Ensuring no more jobs are scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		framework.ExpectError(err)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// only single active job is allowed for ReplaceConcurrent
	ginkgo.It("should replace jobs when ReplaceConcurrent", func() {
		ginkgo.By("Creating a ReplaceConcurrent cronjob")
		cronJob := newTestCronJob("replace", "*/1 * * * ?", batchv1beta1.ReplaceConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to schedule CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to get CronJob %s", cronJob.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Ensuring exactly one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the jobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(activeJobs).To(gomega.HaveLen(1))

		ginkgo.By("Ensuring the job is replaced with a new one")
		err = waitForJobReplaced(f.ClientSet, f.Namespace.Name, jobs.Items[0].Name)
		framework.ExpectNoError(err, "Failed to replace CronJob %s in namespace %s", jobs.Items[0].Name, f.Namespace.Name)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// shouldn't give us unexpected warnings
	ginkgo.It("should not emit unexpected warnings", func() {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1beta1.AllowConcurrent,
			nil, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring at least two jobs and at least one finished job exists by listing jobs explicitly")
		err = waitForJobsAtLeast(f.ClientSet, f.Namespace.Name, 2)
		framework.ExpectNoError(err, "Failed to ensure at least two job exists in namespace %s", f.Namespace.Name)
		err = waitForAnyFinishedJob(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to ensure at least on finished job exists in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring no unexpected event has happened")
		err = waitForEventWithReason(f.ClientSet, f.Namespace.Name, cronJob.Name, []string{"MissingJob", "UnexpectedJob"})
		framework.ExpectError(err)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// deleted jobs should be removed from the active list
	ginkgo.It("should remove from active list jobs that have been deleted", func() {
		ginkgo.By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1beta1.ForbidConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to ensure a %s cronjob is scheduled in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to ensure exactly one %s cronjob is scheduled in namespace %s", cronJob.Name, f.Namespace.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Deleting the job")
		job := cronJob.Status.Active[0]
		framework.ExpectNoError(framework.DeleteResourceAndWaitForGC(f.ClientSet, batchinternal.Kind("Job"), f.Namespace.Name, job.Name))

		ginkgo.By("Ensuring job was deleted")
		_, err = jobutil.GetJob(f.ClientSet, f.Namespace.Name, job.Name)
		framework.ExpectError(err)
		gomega.Expect(errors.IsNotFound(err)).To(gomega.BeTrue())

		ginkgo.By("Ensuring the job is not in the cronjob active list")
		err = waitForJobNotActive(f.ClientSet, f.Namespace.Name, cronJob.Name, job.Name)
		framework.ExpectNoError(err, "Failed to ensure the %s cronjob is not in active list in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring MissingJob event has occurred")
		err = waitForEventWithReason(f.ClientSet, f.Namespace.Name, cronJob.Name, []string{"MissingJob"})
		framework.ExpectNoError(err, "Failed to ensure missing job event has occurred for %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to remove %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// cleanup of successful/failed finished jobs, with successfulJobsHistoryLimit and failedJobsHistoryLimit
	ginkgo.It("should delete successful/failed finished jobs with limit of one job", func() {

		testCases := []struct {
			description  string
			command      []string
			successLimit int32
			failedLimit  int32
		}{
			{
				description:  "successful-jobs-history-limit",
				command:      successCommand,
				successLimit: 1, // keep one successful job
				failedLimit:  0, // keep none failed job
			},
			{
				description:  "failed-jobs-history-limit",
				command:      failureCommand,
				successLimit: 0, // keep none succcessful job
				failedLimit:  1, // keep one failed job
			},
		}

		for _, t := range testCases {
			ginkgo.By(fmt.Sprintf("Creating a AllowConcurrent cronjob with custom %s", t.description))
			cronJob := newTestCronJob(t.description, "*/1 * * * ?", batchv1beta1.AllowConcurrent,
				t.command, &t.successLimit, &t.failedLimit)
			cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
			framework.ExpectNoError(err, "Failed to create allowconcurrent cronjob with custom history limits in namespace %s", f.Namespace.Name)

			// Job is going to complete instantly: do not check for an active job
			// as we are most likely to miss it

			ginkgo.By("Ensuring a finished job exists")
			err = waitForAnyFinishedJob(f.ClientSet, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists in namespace %s", f.Namespace.Name)

			ginkgo.By("Ensuring a finished job exists by listing jobs explicitly")
			jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
			framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists by listing jobs explicitly in namespace %s", f.Namespace.Name)
			_, finishedJobs := filterActiveJobs(jobs)
			framework.ExpectEqual(len(finishedJobs), 1)

			// Job should get deleted when the next job finishes the next minute
			ginkgo.By("Ensuring this job and its pods does not exist anymore")
			err = waitForJobToDisappear(f.ClientSet, f.Namespace.Name, finishedJobs[0])
			framework.ExpectNoError(err, "Failed to ensure that job does not exists anymore in namespace %s", f.Namespace.Name)
			err = waitForJobsPodToDisappear(f.ClientSet, f.Namespace.Name, finishedJobs[0])
			framework.ExpectNoError(err, "Failed to ensure that pods for job does not exists anymore in namespace %s", f.Namespace.Name)

			ginkgo.By("Ensuring there is 1 finished job by listing jobs explicitly")
			jobs, err = f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(metav1.ListOptions{})
			framework.ExpectNoError(err, "Failed to ensure there is one finished job by listing job explicitly in namespace %s", f.Namespace.Name)
			_, finishedJobs = filterActiveJobs(jobs)
			framework.ExpectEqual(len(finishedJobs), 1)

			ginkgo.By("Removing cronjob")
			err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
			framework.ExpectNoError(err, "Failed to remove the %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)
		}
	})
})

// newTestCronJob returns a cronjob which does one of several testing behaviors.
func newTestCronJob(name, schedule string, concurrencyPolicy batchv1beta1.ConcurrencyPolicy,
	command []string, successfulJobsHistoryLimit *int32, failedJobsHistoryLimit *int32) *batchv1beta1.CronJob {
	parallelism := int32(1)
	completions := int32(1)
	backofflimit := int32(1)
	sj := &batchv1beta1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "CronJob",
		},
		Spec: batchv1beta1.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: concurrencyPolicy,
			JobTemplate: batchv1beta1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Parallelism:  &parallelism,
					Completions:  &completions,
					BackoffLimit: &backofflimit,
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
									Image: imageutils.GetE2EImage(imageutils.BusyBox),
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
	sj.Spec.SuccessfulJobsHistoryLimit = successfulJobsHistoryLimit
	sj.Spec.FailedJobsHistoryLimit = failedJobsHistoryLimit
	if command != nil {
		sj.Spec.JobTemplate.Spec.Template.Spec.Containers[0].Command = command
	}
	return sj
}

func createCronJob(c clientset.Interface, ns string, cronJob *batchv1beta1.CronJob) (*batchv1beta1.CronJob, error) {
	return c.BatchV1beta1().CronJobs(ns).Create(cronJob)
}

func getCronJob(c clientset.Interface, ns, name string) (*batchv1beta1.CronJob, error) {
	return c.BatchV1beta1().CronJobs(ns).Get(name, metav1.GetOptions{})
}

func deleteCronJob(c clientset.Interface, ns, name string) error {
	propagationPolicy := metav1.DeletePropagationBackground // Also delete jobs and pods related to cronjob
	return c.BatchV1beta1().CronJobs(ns).Delete(name, &metav1.DeleteOptions{PropagationPolicy: &propagationPolicy})
}

// Wait for at least given amount of active jobs.
func waitForActiveJobs(c clientset.Interface, ns, cronJobName string, active int) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		curr, err := getCronJob(c, ns, cronJobName)
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
		curr, err := getCronJob(c, ns, jobName)
		if err != nil {
			return false, err
		}

		if failIfNonEmpty {
			return len(curr.Status.Active) == 0, nil
		}
		return len(curr.Status.Active) != 0, nil
	})
}

// Wait till a given job actually goes away from the Active list for a given cronjob
func waitForJobNotActive(c clientset.Interface, ns, cronJobName, jobName string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		curr, err := getCronJob(c, ns, cronJobName)
		if err != nil {
			return false, err
		}

		for _, j := range curr.Status.Active {
			if j.Name == jobName {
				return false, nil
			}
		}
		return true, nil
	})
}

// Wait for a job to disappear by listing them explicitly.
func waitForJobToDisappear(c clientset.Interface, ns string, targetJob *batchv1.Job) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		_, finishedJobs := filterActiveJobs(jobs)
		for _, job := range finishedJobs {
			if targetJob.Namespace == job.Namespace && targetJob.Name == job.Name {
				return false, nil
			}
		}
		return true, nil
	})
}

// Wait for a pod to disappear by listing them explicitly.
func waitForJobsPodToDisappear(c clientset.Interface, ns string, targetJob *batchv1.Job) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		options := metav1.ListOptions{LabelSelector: fmt.Sprintf("controller-uid=%s", targetJob.UID)}
		pods, err := c.CoreV1().Pods(ns).List(options)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

// Wait for a job to be replaced with a new one.
func waitForJobReplaced(c clientset.Interface, ns, previousJobName string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		// Ignore Jobs pending deletion, since deletion of Jobs is now asynchronous.
		aliveJobs := filterNotDeletedJobs(jobs)
		if len(aliveJobs) > 1 {
			return false, fmt.Errorf("More than one job is running %+v", jobs.Items)
		} else if len(aliveJobs) == 0 {
			e2elog.Logf("Warning: Found 0 jobs in namespace %v", ns)
			return false, nil
		}
		return aliveJobs[0].Name != previousJobName, nil
	})
}

// waitForJobsAtLeast waits for at least a number of jobs to appear.
func waitForJobsAtLeast(c clientset.Interface, ns string, atLeast int) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(jobs.Items) >= atLeast, nil
	})
}

// waitForAnyFinishedJob waits for any completed job to appear.
func waitForAnyFinishedJob(c clientset.Interface, ns string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(metav1.ListOptions{})
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

// waitForEventWithReason waits for events with a reason within a list has occurred
func waitForEventWithReason(c clientset.Interface, ns, cronJobName string, reasons []string) error {
	return wait.Poll(framework.Poll, 30*time.Second, func() (bool, error) {
		sj, err := getCronJob(c, ns, cronJobName)
		if err != nil {
			return false, err
		}
		events, err := c.CoreV1().Events(ns).Search(scheme.Scheme, sj)
		if err != nil {
			return false, err
		}
		for _, e := range events.Items {
			for _, reason := range reasons {
				if e.Reason == reason {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

// filterNotDeletedJobs returns the job list without any jobs that are pending
// deletion.
func filterNotDeletedJobs(jobs *batchv1.JobList) []*batchv1.Job {
	var alive []*batchv1.Job
	for i := range jobs.Items {
		job := &jobs.Items[i]
		if job.DeletionTimestamp == nil {
			alive = append(alive, job)
		}
	}
	return alive
}

func filterActiveJobs(jobs *batchv1.JobList) (active []*batchv1.Job, finished []*batchv1.Job) {
	for i := range jobs.Items {
		j := jobs.Items[i]
		if !job.IsJobFinished(&j) {
			active = append(active, &j)
		} else {
			finished = append(finished, &j)
		}
	}
	return
}
