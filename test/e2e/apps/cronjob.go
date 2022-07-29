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
	"k8s.io/kubernetes/pkg/controller/job"
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
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	sleepCommand := []string{"sleep", "300"}

	// Pod will complete instantly
	successCommand := []string{"/bin/true"}
	failureCommand := []string{"/bin/false"}

	/*
	   Release: v1.21
	   Testname: CronJob AllowConcurrent
	   Description: CronJob MUST support AllowConcurrent policy, allowing to run multiple jobs at the same time.
	*/
	framework.ConformanceIt("should schedule multiple jobs concurrently", func() {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.AllowConcurrent,
			sleepCommand, nil, nil)
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring more than one job is running at a time")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 2)
		framework.ExpectNoError(err, "Failed to wait for active jobs in CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring at least two running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(len(activeJobs)).To(gomega.BeNumerically(">=", 2))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	/*
	   Release: v1.21
	   Testname: CronJob Suspend
	   Description: CronJob MUST support suspension, which suppresses creation of new jobs.
	*/
	framework.ConformanceIt("should not schedule jobs when suspended [Slow]", func() {
		ginkgo.By("Creating a suspended cronjob")
		cronJob := newTestCronJob("suspended", "*/1 * * * ?", batchv1.AllowConcurrent,
			sleepCommand, nil, nil)
		t := true
		cronJob.Spec.Suspend = &t
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring no jobs are scheduled")
		err = waitForNoJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, false)
		framework.ExpectError(err)

		ginkgo.By("Ensuring no job exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		gomega.Expect(jobs.Items).To(gomega.HaveLen(0))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	/*
	   Release: v1.21
	   Testname: CronJob ForbidConcurrent
	   Description: CronJob MUST support ForbidConcurrent policy, allowing to run single, previous job at the time.
	*/
	framework.ConformanceIt("should not schedule new jobs when ForbidConcurrent [Slow]", func() {
		ginkgo.By("Creating a ForbidConcurrent cronjob")
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1.ForbidConcurrent,
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
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
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

	/*
	   Release: v1.21
	   Testname: CronJob ReplaceConcurrent
	   Description: CronJob MUST support ReplaceConcurrent policy, allowing to run single, newer job at the time.
	*/
	framework.ConformanceIt("should replace jobs when ReplaceConcurrent", func() {
		ginkgo.By("Creating a ReplaceConcurrent cronjob")
		cronJob := newTestCronJob("replace", "*/1 * * * ?", batchv1.ReplaceConcurrent,
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
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
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

	ginkgo.It("should be able to schedule after more than 100 missed schedule", func() {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.ForbidConcurrent,
			sleepCommand, nil, nil)
		creationTime := time.Now().Add(-99 * 24 * time.Hour)
		lastScheduleTime := creationTime.Add(1 * 24 * time.Hour)
		cronJob.CreationTimestamp = metav1.Time{Time: creationTime}
		cronJob.Status.LastScheduleTime = &metav1.Time{Time: lastScheduleTime}
		cronJob, err := createCronJob(f.ClientSet, f.Namespace.Name, cronJob)
		framework.ExpectNoError(err, "Failed to create CronJob in namespace %s", f.Namespace.Name)

		ginkgo.By("Ensuring one job is running")
		err = waitForActiveJobs(f.ClientSet, f.Namespace.Name, cronJob.Name, 1)
		framework.ExpectNoError(err, "Failed to wait for active jobs in CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)

		ginkgo.By("Ensuring at least one running jobs exists by listing jobs explicitly")
		jobs, err := f.ClientSet.BatchV1().Jobs(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list the CronJobs in namespace %s", f.Namespace.Name)
		activeJobs, _ := filterActiveJobs(jobs)
		gomega.Expect(len(activeJobs)).To(gomega.BeNumerically(">=", 1))

		ginkgo.By("Removing cronjob")
		err = deleteCronJob(f.ClientSet, f.Namespace.Name, cronJob.Name)
		framework.ExpectNoError(err, "Failed to delete CronJob %s in namespace %s", cronJob.Name, f.Namespace.Name)
	})

	// shouldn't give us unexpected warnings
	ginkgo.It("should not emit unexpected warnings", func() {
		ginkgo.By("Creating a cronjob")
		cronJob := newTestCronJob("concurrent", "*/1 * * * ?", batchv1.AllowConcurrent,
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
		cronJob := newTestCronJob("forbid", "*/1 * * * ?", batchv1.ForbidConcurrent,
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
		framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(f.ClientSet, batchinternal.Kind("Job"), f.Namespace.Name, job.Name))

		ginkgo.By("Ensuring job was deleted")
		_, err = e2ejob.GetJob(f.ClientSet, f.Namespace.Name, job.Name)
		framework.ExpectError(err)
		framework.ExpectEqual(apierrors.IsNotFound(err), true)

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

	// cleanup of successful finished jobs, with limit of one successful job
	ginkgo.It("should delete successful finished jobs with limit of one successful job", func() {
		ginkgo.By("Creating an AllowConcurrent cronjob with custom history limit")
		successLimit := int32(1)
		failedLimit := int32(0)
		cronJob := newTestCronJob("successful-jobs-history-limit", "*/1 * * * ?", batchv1.AllowConcurrent,
			successCommand, &successLimit, &failedLimit)

		ensureHistoryLimits(f.ClientSet, f.Namespace.Name, cronJob)
	})

	// cleanup of failed finished jobs, with limit of one failed job
	ginkgo.It("should delete failed finished jobs with limit of one job", func() {
		ginkgo.By("Creating an AllowConcurrent cronjob with custom history limit")
		successLimit := int32(0)
		failedLimit := int32(1)
		cronJob := newTestCronJob("failed-jobs-history-limit", "*/1 * * * ?", batchv1.AllowConcurrent,
			failureCommand, &successLimit, &failedLimit)

		ensureHistoryLimits(f.ClientSet, f.Namespace.Name, cronJob)
	})

	/*
	   Release: v1.21
	   Testname: CronJob API Operations
	   Description:
	   CronJob MUST support create, get, list, watch, update, patch, delete, and deletecollection.
	   CronJob/status MUST support get, update and patch.
	*/
	framework.ConformanceIt("should support CronJob API operations", func() {
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
		createdCronJob, err := cjClient.Create(context.TODO(), cjTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenCronJob, err := cjClient.Get(context.TODO(), createdCronJob.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gottenCronJob.UID, createdCronJob.UID)

		ginkgo.By("listing")
		cjs, err := cjClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(cjs.Items), 1, "filtered list should have 1 item")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		cjWatch, err := cjClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: cjs.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		// Test cluster-wide list and watch
		clusterCJClient := f.ClientSet.BatchV1().CronJobs("")
		ginkgo.By("cluster-wide listing")
		clusterCJs, err := clusterCJClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(clusterCJs.Items), 1, "filtered list should have 1 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterCJClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: cjs.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedCronJob, err := cjClient.Patch(context.TODO(), createdCronJob.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedCronJob.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		var cjToUpdate, updatedCronJob *batchv1.CronJob
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			cjToUpdate, err = cjClient.Get(context.TODO(), createdCronJob.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			cjToUpdate.Annotations["updated"] = "true"
			updatedCronJob, err = cjClient.Update(context.TODO(), cjToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedCronJob.Annotations["updated"], "true", "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-cjWatch.ResultChan():
				framework.ExpectEqual(ok, true, "watch channel should not close")
				framework.ExpectEqual(evt.Type, watch.Modified)
				watchedCronJob, isCronJob := evt.Object.(*batchv1.CronJob)
				framework.ExpectEqual(isCronJob, true, fmt.Sprintf("expected CronJob, got %T", evt.Object))
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
		patchedStatus, err := cjClient.Patch(context.TODO(), createdCronJob.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":`+string(cjStatusJSON)+`}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedStatus.Status.LastScheduleTime.Equal(&now1), true, "patched object should have the applied lastScheduleTime status")
		framework.ExpectEqual(patchedStatus.Annotations["patchedstatus"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		// we need to use RFC3339 version since conversion over the wire cuts nanoseconds
		now2 := metav1.Now().Rfc3339Copy()
		var statusToUpdate, updatedStatus *batchv1.CronJob
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = cjClient.Get(context.TODO(), createdCronJob.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			statusToUpdate.Status.LastScheduleTime = &now2
			updatedStatus, err = cjClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedStatus.Status.LastScheduleTime.Equal(&now2), true, fmt.Sprintf("updated object status expected to have updated lastScheduleTime %#v, got %#v", statusToUpdate.Status.LastScheduleTime, updatedStatus.Status.LastScheduleTime))

		ginkgo.By("get /status")
		cjResource := schema.GroupVersionResource{Group: "batch", Version: cjVersion, Resource: "cronjobs"}
		gottenStatus, err := f.DynamicClient.Resource(cjResource).Namespace(ns).Get(context.TODO(), createdCronJob.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		statusUID, _, err := unstructured.NestedFieldCopy(gottenStatus.Object, "metadata", "uid")
		framework.ExpectNoError(err)
		framework.ExpectEqual(string(createdCronJob.UID), statusUID, fmt.Sprintf("createdCronJob.UID: %v expected to match statusUID: %v ", createdCronJob.UID, statusUID))

		// CronJob resource delete operations
		expectFinalizer := func(cj *batchv1.CronJob, msg string) {
			framework.ExpectNotEqual(cj.DeletionTimestamp, nil, fmt.Sprintf("expected deletionTimestamp, got nil on step: %q, cronjob: %+v", msg, cj))
			framework.ExpectEqual(len(cj.Finalizers) > 0, true, fmt.Sprintf("expected finalizers on cronjob, got none on step: %q, cronjob: %+v", msg, cj))
		}

		ginkgo.By("deleting")
		cjTemplate.Name = "for-removal"
		forRemovalCronJob, err := cjClient.Create(context.TODO(), cjTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		err = cjClient.Delete(context.TODO(), forRemovalCronJob.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		cj, err := cjClient.Get(context.TODO(), forRemovalCronJob.Name, metav1.GetOptions{})
		// If controller does not support finalizers, we expect a 404.  Otherwise we validate finalizer behavior.
		if err == nil {
			expectFinalizer(cj, "deleting cronjob")
		} else {
			framework.ExpectEqual(apierrors.IsNotFound(err), true, fmt.Sprintf("expected 404, got %v", err))
		}

		ginkgo.By("deleting a collection")
		err = cjClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		cjs, err = cjClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 2 items since some cronjobs might not have been deleted yet due to finalizers
		framework.ExpectEqual(len(cjs.Items) <= 2, true, "filtered list should be <= 2")
		// Validate finalizers
		for _, cj := range cjs.Items {
			expectFinalizer(&cj, "deleting cronjob collection")
		}
	})

})

func ensureHistoryLimits(c clientset.Interface, ns string, cronJob *batchv1.CronJob) {
	cronJob, err := createCronJob(c, ns, cronJob)
	framework.ExpectNoError(err, "Failed to create allowconcurrent cronjob with custom history limits in namespace %s", ns)

	// Job is going to complete instantly: do not check for an active job
	// as we are most likely to miss it

	ginkgo.By("Ensuring a finished job exists")
	err = waitForAnyFinishedJob(c, ns)
	framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists in namespace %s", ns)

	ginkgo.By("Ensuring a finished job exists by listing jobs explicitly")
	jobs, err := c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to ensure a finished cronjob exists by listing jobs explicitly in namespace %s", ns)
	activeJobs, finishedJobs := filterActiveJobs(jobs)
	if len(finishedJobs) != 1 {
		framework.Logf("Expected one finished job in namespace %s; activeJobs=%v; finishedJobs=%v", ns, activeJobs, finishedJobs)
		framework.ExpectEqual(len(finishedJobs), 1)
	}

	// Job should get deleted when the next job finishes the next minute
	ginkgo.By("Ensuring this job and its pods does not exist anymore")
	err = waitForJobToDisappear(c, ns, finishedJobs[0])
	framework.ExpectNoError(err, "Failed to ensure that job does not exists anymore in namespace %s", ns)
	err = waitForJobsPodToDisappear(c, ns, finishedJobs[0])
	framework.ExpectNoError(err, "Failed to ensure that pods for job does not exists anymore in namespace %s", ns)

	ginkgo.By("Ensuring there is 1 finished job by listing jobs explicitly")
	jobs, err = c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to ensure there is one finished job by listing job explicitly in namespace %s", ns)
	activeJobs, finishedJobs = filterActiveJobs(jobs)
	if len(finishedJobs) != 1 {
		framework.Logf("Expected one finished job in namespace %s; activeJobs=%v; finishedJobs=%v", ns, activeJobs, finishedJobs)
		framework.ExpectEqual(len(finishedJobs), 1)
	}

	ginkgo.By("Removing cronjob")
	err = deleteCronJob(c, ns, cronJob.Name)
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

func createCronJob(c clientset.Interface, ns string, cronJob *batchv1.CronJob) (*batchv1.CronJob, error) {
	return c.BatchV1().CronJobs(ns).Create(context.TODO(), cronJob, metav1.CreateOptions{})
}

func getCronJob(c clientset.Interface, ns, name string) (*batchv1.CronJob, error) {
	return c.BatchV1().CronJobs(ns).Get(context.TODO(), name, metav1.GetOptions{})
}

func deleteCronJob(c clientset.Interface, ns, name string) error {
	propagationPolicy := metav1.DeletePropagationBackground // Also delete jobs and pods related to cronjob
	return c.BatchV1().CronJobs(ns).Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &propagationPolicy})
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
		jobs, err := c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
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
		pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

// Wait for a job to be replaced with a new one.
func waitForJobReplaced(c clientset.Interface, ns, previousJobName string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
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
func waitForJobsAtLeast(c clientset.Interface, ns string, atLeast int) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(jobs.Items) >= atLeast, nil
	})
}

// waitForAnyFinishedJob waits for any completed job to appear.
func waitForAnyFinishedJob(c clientset.Interface, ns string) error {
	return wait.Poll(framework.Poll, cronJobTimeout, func() (bool, error) {
		jobs, err := c.BatchV1().Jobs(ns).List(context.TODO(), metav1.ListOptions{})
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
