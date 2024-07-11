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
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	jobutil "k8s.io/kubernetes/pkg/controller/job/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// How long to wait for a cronjob
	cronJobTimeout = 5 * time.Minute
)

var _ = SIGDescribe("CronJob", func() {
	f := framework.NewDefaultFramework("cronjob")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	sleepCommand := []string{"sleep", "300"}

	// Pod will complete instantly
	successCommand := []string{"/bin/true"}
	failureCommand := []string{"/bin/false"}

	/*
	   Release: v1.21
	   Testname: CronJob AllowConcurrent
	   Description: CronJob MUST support AllowConcurrent policy, allowing to run multiple jobs at the same time.
	*/
	framework.ConformanceIt("should schedule multiple jobs concurrently", func(ctx context.Context) {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.AllowConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring more than one job is running at a time")
		err = waitForActiveJobs(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		framework.ExpectNoError(err, "Failed to wait for active jobs in CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring at least two running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(len(activeJobs)).To(gomega.BeNumerically(">=", 2))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	/*
	   Release: v1.21
	   Testname: CronJob Suspend
	   Description: CronJob MUST support suspension, which suppresses creation of new jobs.
	*/
	framework.ConformanceIt("should not schedule jobs when suspended", f.WithSlow(), func(ctx context.Context) {
		ginkgo.By("Creating a suspended cronjob")
		cronJob := newTestCronJob("suspended", "*/1 * * * ?", batchv1.AllowConcurrent,
			sleepCommand, nil, nil)
		t := true
		cronJob.Spec.Suspend = &t
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring no jobs are scheduled")
		gomega.Consistently(ctx, framework.GetObject(f.ClientSet.BatchV1().CronJobs(f.Namespace.Name).Get, cronJob.Name, metav1.GetOptions{})).WithPolling(framework.Poll).WithTimeout(cronJobTimeout).
			Should(gomega.HaveField("Status.Active", gomega.BeEmpty()))

		ginkgo.By("Ensuring no job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		gomega.Expect(jobs.Items).To(gomega.BeEmpty())

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	/*
	   Release: v1.21
	   Testname: CronJob ForbidConcurrent
	   Description: CronJob MUST support ForbidConcurrent policy, allowing to run single, previous job at the time.
	*/
	framework.ConformanceIt("should not schedule new jobs when ForbidConcurrent", f.WithSlow(), func(ctx context.Context) {
		ginkgo.By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1.ForbidConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to schedule CronJob %s", cronJob.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to get CronJob %s", cronJob.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Ensuring exactly one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(activeJobs).To(gomega.HaveLen(1))

		ginkgo.By("Ensuring no more jobs are scheduled")
		gomega.Eventually(ctx, framework.GetObject(f.ClientSet.BatchV1().CronJobs(f.Namespace.Name).Get, cronJob.Name, metav1.GetOptions{})).WithPolling(framework.Poll).WithTimeout(cronJobTimeout).
			Should(framework.MakeMatcher(func(cj *batchv1.CronJob) (func() string, error) {
				if len(cj.Status.Active) < 2 {
					return nil, nil
				}
				return func() string {
					return fmt.Sprintf("unexpect active job number: %d\n", len(cj.Status.Active))
				}, nil
			}))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	/*
	   Release: v1.21
	   Testname: CronJob ReplaceConcurrent
	   Description: CronJob MUST support ReplaceConcurrent policy, allowing to run single, newer job at the time.
	*/
	framework.ConformanceIt("should replace jobs when ReplaceConcurrent", func(ctx context.Context) {
		ginkgo.By("Creating a ReplaceConcurrent cronjob")
		cronJob := newTestCronJob("replace", "*/1 * * * ?", batchv1.ReplaceConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to schedule CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to get CronJob %s", cronJob.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Ensuring exactly one running job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the jobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(activeJobs).To(gomega.HaveLen(1))

		ginkgo.By("Ensuring the job is replaced with a new one")
		err = waitForJobReplaced(ctx, f.ClientSet, f.Namespace.Name, jobs.Items[0].Name)
		framework.ExpectNoError(err, "Failed to replace CronJob %s in namespace %s", jobs.Items[0].Name, f.Namespace.Name)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	ginkgo.It("should be able to schedule after more than 100 missed schedule", func(ctx context.Context) {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.ForbidConcurrent,
			sleepCommand, nil, nil)
		creationTime := time.Now().Add(-99 * 24 * time.Hour)
		lastScheduleTime := creationTime.Add(1 * 24 * time.Hour)
		cronJob.CreationTimestamp = metav1.Time{Time: creationTime}
		cronJob.Status.LastScheduleTime = &metav1.Time{Time: lastScheduleTime}
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring one job is running")
		err = waitForActiveJobs(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to wait for active jobs in CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring at least one running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(activeJobs).ToNot(gomega.BeEmpty())

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// shouldn't give us unexpected warnings
	ginkgo.It("should not emit unexpected warnings", func(ctx context.Context) {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.AllowConcurrent,
			nil, nil, nil)
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring at least two jobs and at least one finished job exists by listing jobs explicitly")
		err = waitForJobsAtLeast(ctx, f.ClientSet, f.Namespace.Name, 2)
		framework.ExpectNoError(err, "Failed to ensure at least two job exists in namespace %s", f.Namespace.Name)
		err = waitForAnyFinishedJob(ctx, f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to ensure at least on finished job exists in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring no unexpected event has happened")
		gomega.Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*v1.EventList, error) {
			sj, err := getCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
			if err != nil {
				return nil, err
			}
			return f.ClientSet.CoreV1().Events(f.Namespace.Name).Search(scheme.Scheme, sj)
		})).WithPolling(framework.Poll).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(actual *v1.EventList) (failure func() string, err error) {
			for _, e := range actual.Items {
				for _, reason := range []string{"MissingJob", "UnexpectedJob"} {
					if e.Reason == reason {
						return func() string {
							return fmt.Sprintf("unexpected event: %s\n", reason)
						}, nil
					}
				}
			}
			return nil, nil
		}))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// deleted jobs should be removed from the active list
	ginkgo.It("should remove from active list jobs that have been deleted", func(ctx context.Context) {
		ginkgo.By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1.ForbidConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring a job is scheduled")
		err = waitForActiveJobs(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to ensure a %s cronjob is scheduled in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring exactly one is scheduled")
		cronJob, err = getCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to ensure exactly one %s cronjob is scheduled in namespace %s", cronJob.Name, f.Namespace.Name)
		gomega.Expect(cronJob.Status.Active).Should(gomega.HaveLen(1))

		ginkgo.By("Deleting the job")
		job := cronJob.Status.Active[0]
		framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(ctx, f.ClientSet, batchinternal.Kind("Job"), f.Namespace.Name, job.Name))

		ginkgo.By("Ensuring job was deleted")
		_, err = e2ejob.GetJob(ctx, f.ClientSet, f.Namespace.Name, job.Name)
		gomega.Expect(err).To(gomega.MatchError(apierrors.IsNotFound, fmt.Sprintf("Failed to delete %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)))

		ginkgo.By("Ensuring the job is not in the cronjob active list")
		err = waitForJobNotActive(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, job.Name)
		framework.ExpectNoError(err, "Failed to ensure the %s cronjob is not in active list in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring MissingJob event has occurred")
		err = waitForEventWithReason(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name, []string{"MissingJob"})
		framework.ExpectNoError(err, "Failed to ensure missing job event has occurred for %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to remove %s cronjob in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// cleanup of successful finished jobs, with limit of one successful job
	ginkgo.It("should delete successful finished jobs with limit of one successful job", func(ctx context.Context) {
		ginkgo.By("Creating an AllowConcurrent cronjob with custom history limit")
		successLimit := int32(1)
		failedLimit := int32(0)
		cronJob := newTestCronJob("successful-jobs-history-limit", "*/1 * * * ?", batchv1.AllowConcurrent,
			successCommand, &successLimit, &failedLimit)

		ensureHistoryLimits(ctx, f.ClientSet, f.Namespace.Name, cronJob)
	})

	// cleanup of failed finished jobs, with limit of one failed job
	ginkgo.It("should delete failed finished jobs with limit of one job", func(ctx context.Context) {
		ginkgo.By("Creating an AllowConcurrent cronjob with custom history limit")
		successLimit := int32(0)
		failedLimit := int32(1)
		cronJob := newTestCronJob("failed-jobs-history-limit", "*/1 * * * ?", batchv1.AllowConcurrent,
			failureCommand, &successLimit, &failedLimit)

		ensureHistoryLimits(ctx, f.ClientSet, f.Namespace.Name, cronJob)
	})

	ginkgo.It("should support timezone", func(ctx context.Context) {
		ginkgo.By("Creating a cronjob with TimeZone")
		cronJob := newTestCronJob("cronjob-with-timezone", "*/1 * * * ?", batchv1.AllowConcurrent,
			failureCommand, nil, nil)
		badTimeZone := "bad-time-zone"
		cronJob.Spec.TimeZone = &badTimeZone
		_, err := createCronJob(ctx, f.ClientSet, f.Namespace.Name, cronJob)
		gomega.Expect(err).To(gomega.MatchError(apierrors.IsInvalid, "Failed to create CronJob, invalid time zone."))
	})

	/*
	   Release: v1.21
	   Testname: CronJob API Operations
	   Description:
	   CronJob MUST support create, get, list, watch, update, patch, delete, and deletecollection.
	   CronJob/status MUST support get, update and patch.
	*/
	framework.ConformanceIt("should support CronJob API operations", func(ctx context.Context) {
		ginkgo.By("Creating a cronjob")
		successLimit := int32(1)
		failedLimit := int32(0)
		cjTemplate := newTestCronJob("test-api", "* */1 * * ?", batchv1.AllowConcurrent,
			successCommand, &successLimit, &failedLimit)
		cjTemplate.Labels = map[string]string{
			"special-label": f.UniqueName,
		}

		ns := f.Namespace.Name
		cjVersion := "v1"
		cjClient := f.ClientSet.BatchV1().CronJobs(ns)

		ginkgo.By("creating")
		createdCronJob, err := cjClient.Create(ctx, cjTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenCronJob, err := cjClient.Get(ctx, createdCronJob.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenCronJob.UID).To(gomega.Equal(createdCronJob.UID))

		ginkgo.By("listing")
		cjs, err := cjClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(cjs.Items).To(gomega.HaveLen(1), "filtered list should have 1 item")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		cjWatch, err := cjClient.Watch(ctx, metav1.ListOptions{ResourceVersion: cjs.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		// Test cluster-wide list and watch
		clusterCJClient := f.ClientSet.BatchV1().CronJobs("")
		ginkgo.By("cluster-wide listing")
		clusterCJs, err := clusterCJClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(clusterCJs.Items).To(gomega.HaveLen(1), "filtered list should have 1 item")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterCJClient.Watch(ctx, metav1.ListOptions{ResourceVersion: cjs.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedCronJob, err := cjClient.Patch(ctx, createdCronJob.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedCronJob.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		var cjToUpdate, updatedCronJob *batchv1.CronJob
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			cjToUpdate, err = cjClient.Get(ctx, createdCronJob.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			cjToUpdate.Annotations["updated"] = "true"
			updatedCronJob, err = cjClient.Update(ctx, cjToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		gomega.Expect(updatedCronJob.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-cjWatch.ResultChan():

				if !ok {
					framework.Fail("Watch channel is closed.")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedCronJob, isCronJob := evt.Object.(*batchv1.CronJob)
				if !isCronJob {
					framework.Failf("expected CronJob, got %T", evt.Object)
				}
				if watchedCronJob.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					cjWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedCronJob.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		// /status subresource operations
		ginkgo.By("patching /status")
		// we need to use RFC3339 version since conversion over the wire cuts nanoseconds
		now1 := metav1.Now().Rfc3339Copy()
		cjStatus := batchv1.CronJobStatus{
			LastScheduleTime: &now1,
		}
		cjStatusJSON, err := json.Marshal(cjStatus)
		framework.ExpectNoError(err)
		patchedStatus, err := cjClient.Patch(ctx, createdCronJob.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":`+string(cjStatusJSON)+`}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		if !patchedStatus.Status.LastScheduleTime.Equal(&now1) {
			framework.Failf("patched object should have the applied lastScheduleTime %#v, got %#v instead", cjStatus.LastScheduleTime, patchedStatus.Status.LastScheduleTime)
		}
		gomega.Expect(patchedStatus.Annotations).To(gomega.HaveKeyWithValue("patchedstatus", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		// we need to use RFC3339 version since conversion over the wire cuts nanoseconds
		now2 := metav1.Now().Rfc3339Copy()
		var statusToUpdate, updatedStatus *batchv1.CronJob
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = cjClient.Get(ctx, createdCronJob.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			statusToUpdate.Status.LastScheduleTime = &now2
			updatedStatus, err = cjClient.UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)

		if !updatedStatus.Status.LastScheduleTime.Equal(&now2) {
			framework.Failf("updated object status expected to have updated lastScheduleTime %#v, got %#v", statusToUpdate.Status.LastScheduleTime, updatedStatus.Status.LastScheduleTime)
		}

		ginkgo.By("get /status")
		cjResource := schema.GroupVersionResource{Group: "batch", Version: cjVersion, Resource: "cronjobs"}
		gottenStatus, err := f.DynamicClient.Resource(cjResource).Namespace(ns).Get(ctx, createdCronJob.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		statusUID, _, err := unstructured.NestedFieldCopy(gottenStatus.Object, "metadata", "uid")
		framework.ExpectNoError(err)
		gomega.Expect(string(createdCronJob.UID)).To(gomega.Equal(statusUID), "createdCronJob.UID: %v expected to match statusUID: %v ", createdCronJob.UID, statusUID)

		// CronJob resource delete operations
		expectFinalizer := func(cj *batchv1.CronJob, msg string) {
			gomega.Expect(cj.DeletionTimestamp).NotTo(gomega.BeNil(), fmt.Sprintf("expected deletionTimestamp, got nil on step: %q, cronjob: %+v", msg, cj))
			gomega.Expect(cj.Finalizers).ToNot(gomega.BeEmpty(), "expected finalizers on cronjob, got none on step: %q, cronjob: %+v", msg, cj)
		}

		ginkgo.By("deleting")
		cjTemplate.Name = "for-removal"
		forRemovalCronJob, err := cjClient.Create(ctx, cjTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		err = cjClient.Delete(ctx, forRemovalCronJob.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		cj, err := cjClient.Get(ctx, forRemovalCronJob.Name, metav1.GetOptions{})
		// If controller does not support finalizers, we expect a 404.  Otherwise we validate finalizer behavior.
		if err == nil {
			expectFinalizer(cj, "deleting cronjob")
		} else if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %v", err)
		}

		ginkgo.By("deleting a collection")
		err = cjClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		cjs, err = cjClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 2 items since some cronjobs might not have been deleted yet due to finalizers
		gomega.Expect(len(cjs.Items)).To(gomega.BeNumerically("<=", 2), "filtered list length should be <= 2, got:\n%s", format.Object(cjs.Items, 1))
		// Validate finalizers
		for _, cj := range cjs.Items {
			expectFinalizer(&cj, "deleting cronjob collection")
		}
	})

})

func ensureHistoryLimits(ctx context.Context, c clientset.Interface, ns string, cronJob *batchv1.CronJob) {
	cronJob, err := createCronJob(ctx, c, ns, cronJob)
	framework.ExpectNoError(err, "Failed to create allowconcurrent cronjob with custom history limits in namespace %s", ns)

	// Job is going to complete instantly: do not check for an active job
	// as we are most likely to miss it

	ginkgo.By("Ensuring a finished job exists")
	err = waitForAnyFinishedJob(ctx, c, ns)
	framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists in namespace %s", ns)

	ginkgo.By("Ensuring a finished job exists by listing jobs explicitly")
	jobs, err := c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists by listing jobs explicitly in namespace %s", ns)
	activeJobs, finishedJobs := filterActiveJobs(jobs)
	if len(finishedJobs) != 1 {
		framework.Logf("Expected one finished job in namespace %s; activeJobs=%v; finishedJobs=%v", ns, activeJobs, finishedJobs)
		gomega.Expect(finishedJobs).To(gomega.HaveLen(1))
	}

	// Job should get deleted when the next job finishes the next minute
	ginkgo.By("Ensuring this job and its pods does not exist anymore")
	err = waitForJobToDisappear(ctx, c, ns, finishedJobs[0])
	framework.ExpectNoError(err, "Failed to ensure that job does not exists anymore in namespace %s", ns)
	err = waitForJobsPodToDisappear(ctx, c, ns, finishedJobs[0])
	framework.ExpectNoError(err, "Failed to ensure that pods for job does not exists anymore in namespace %s", ns)

	ginkgo.By("Ensuring there is 1 finished job by listing jobs explicitly")
	jobs, err = c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to ensure there is one finished job by listing job explicitly in namespace %s", ns)
	activeJobs, finishedJobs = filterActiveJobs(jobs)
	if len(finishedJobs) != 1 {
		framework.Logf("Expected one finished job in namespace %s; activeJobs=%v; finishedJobs=%v", ns, activeJobs, finishedJobs)
		gomega.Expect(finishedJobs).To(gomega.HaveLen(1))
	}

	ginkgo.By("Removing cronjob")
	err = deleteCronJob(ctx, c, ns, cronJob.Name)
	framework.ExpectNoError(err, "Failed to remove the %s cronjob in namespace %s", cronJob.Name, ns)
}

// newTestCronJob returns a cronjob which does one of several testing behaviors.
func newTestCronJob(name, schedule string, concurrencyPolicy batchv1.ConcurrencyPolicy,
	command []string, successfulJobsHistoryLimit *int32, failedJobsHistoryLimit *int32) *batchv1.CronJob {
	parallelism := int32(1)
	completions := int32(1)
	backofflimit := int32(1)
	sj := &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "CronJob",
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: concurrencyPolicy,
			JobTemplate: batchv1.JobTemplateSpec{
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

func createCronJob(ctx context.Context, c clientset.Interface, ns string, cronJob *batchv1.CronJob) (*batchv1.CronJob, error) {
	return c.BatchV1().CronJobs(ns).Create(ctx, cronJob, metav1.CreateOptions{})
}

func getCronJob(ctx context.Context, c clientset.Interface, ns, name string) (*batchv1.CronJob, error) {
	return c.BatchV1().CronJobs(ns).Get(ctx, name, metav1.GetOptions{})
}

func deleteCronJob(ctx context.Context, c clientset.Interface, ns, name string) error {
	propagationPolicy := metav1.DeletePropagationBackground // Also delete jobs and pods related to cronjob
	return c.BatchV1().CronJobs(ns).Delete(ctx, name, metav1.DeleteOptions{PropagationPolicy: &propagationPolicy})
}

// Wait for at least given amount of active jobs.
func waitForActiveJobs(ctx context.Context, c clientset.Interface, ns, cronJobName string, active int) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		curr, err := getCronJob(ctx, c, ns, cronJobName)
		if err != nil {
			return false, err
		}
		return len(curr.Status.Active) >= active, nil
	})
}

// Wait till a given job actually goes away from the Active list for a given cronjob
func waitForJobNotActive(ctx context.Context, c clientset.Interface, ns, cronJobName, jobName string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		curr, err := getCronJob(ctx, c, ns, cronJobName)
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
func waitForJobToDisappear(ctx context.Context, c clientset.Interface, ns string, targetJob *batchv1.Job) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
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
func waitForJobsPodToDisappear(ctx context.Context, c clientset.Interface, ns string, targetJob *batchv1.Job) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		options := metav1.ListOptions{LabelSelector: fmt.Sprintf("controller-uid=%s", targetJob.UID)}
		pods, err := c.CoreV1().Pods(ns).List(ctx, options)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

// Wait for a job to be replaced with a new one.
func waitForJobReplaced(ctx context.Context, c clientset.Interface, ns, previousJobName string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		// Ignore Jobs pending deletion, since deletion of Jobs is now asynchronous.
		aliveJobs := filterNotDeletedJobs(jobs)
		if len(aliveJobs) > 1 {
			return false, fmt.Errorf("more than one job is running %+v", jobs.Items)
		} else if len(aliveJobs) == 0 {
			framework.Logf("Warning: Found 0 jobs in namespace %v", ns)
			return false, nil
		}
		return aliveJobs[0].Name != previousJobName, nil
	})
}

// waitForJobsAtLeast waits for at least a number of jobs to appear.
func waitForJobsAtLeast(ctx context.Context, c clientset.Interface, ns string, atLeast int) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(jobs.Items) >= atLeast, nil
	})
}

// waitForAnyFinishedJob waits for any completed job to appear.
func waitForAnyFinishedJob(ctx context.Context, c clientset.Interface, ns string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, cronJobTimeout, false, func(ctx context.Context) (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		for i := range jobs.Items {
			if jobutil.IsJobFinished(&jobs.Items[i]) {
				return true, nil
			}
		}
		return false, nil
	})
}

// waitForEventWithReason waits for events with a reason within a list has occurred
func waitForEventWithReason(ctx context.Context, c clientset.Interface, ns, cronJobName string, reasons []string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		sj, err := getCronJob(ctx, c, ns, cronJobName)
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
		if !jobutil.IsJobFinished(&j) {
			active = append(active, &j)
		} else {
			finished = append(finished, &j)
		}
	}
	return
}
