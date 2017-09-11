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

package e2e_federation

import (
	"fmt"
	"strings"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api"
)

const (
	FederationJobName = "federation-job"
)

var _ = framework.KubeDescribe("Federation jobs [Feature:Federation]", func() {

	f := fedframework.NewDefaultFederatedFramework("federation-job")

	Describe("Job objects [NoCluster]", func() {
		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// Delete all jobs.
			nsName := f.FederationNamespace.Name
			deleteAllJobsOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			job := createJobOrFail(f.FederationClientset, nsName)
			By(fmt.Sprintf("Creation of job %q in namespace %q succeeded.  Deleting job.", job.Name, nsName))
			// Cleanup
			err := f.FederationClientset.Batch().Jobs(nsName).Delete(job.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting job %q in namespace %q", job.Name, job.Namespace)
			By(fmt.Sprintf("Deletion of job %q in namespace %q succeeded.", job.Name, nsName))
		})

	})

	// e2e cases for federated job controller
	Describe("Federated Job", func() {
		var (
			clusters fedframework.ClusterSlice
		)
		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters = f.GetRegisteredClusters()
		})

		AfterEach(func() {
			nsName := f.FederationNamespace.Name
			deleteAllJobsOrFail(f.FederationClientset, nsName)
		})

		It("should create and update matching jobs in underlying clusters", func() {
			nsName := f.FederationNamespace.Name
			job := createJobOrFail(f.FederationClientset, nsName)
			defer func() {
				// cleanup. deletion of jobs is not supported for underlying clusters
				By(fmt.Sprintf("Deleting job %q/%q", nsName, job.Name))
				waitForJobOrFail(f.FederationClientset, nsName, job.Name, clusters)
				f.FederationClientset.Batch().Jobs(nsName).Delete(job.Name, &metav1.DeleteOptions{})
			}()

			waitForJobOrFail(f.FederationClientset, nsName, job.Name, clusters)
			By(fmt.Sprintf("Successfuly created and synced job %q/%q to clusters", nsName, job.Name))
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForJob(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that jobs were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForJob(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that jobs were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForJob(f.FederationClientset, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that jobs were not deleted from underlying clusters"))
		})

	})
})

// deleteAllJobsOrFail deletes all jobs in the given namespace name.
func deleteAllJobsOrFail(clientset *fedclientset.Clientset, nsName string) {
	jobList, err := clientset.Batch().Jobs(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, job := range jobList.Items {
		deleteJobOrFail(clientset, nsName, job.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForJob verifies that job are deleted
// from underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForJob(clientset *fedclientset.Clientset, clusters fedframework.ClusterSlice, orphanDependents *bool, nsName string) {
	job := createJobOrFail(clientset, nsName)
	jobName := job.Name
	// Check subclusters if the job was created there.
	By(fmt.Sprintf("Waiting for job %s to be created in all underlying clusters", jobName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Batch().Jobs(nsName).Get(jobName, metav1.GetOptions{})
			if err != nil && errors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all jobs created")

	By(fmt.Sprintf("Deleting job %s", jobName))
	deleteJobOrFail(clientset, nsName, jobName, orphanDependents)

	By(fmt.Sprintf("Verifying job %s in underlying clusters", jobName))
	errMessages := []string{}
	// job should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for _, cluster := range clusters {
		clusterName := cluster.Name
		_, err := cluster.Batch().Jobs(nsName).Get(jobName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for job %s in cluster %s, expected job to exist", jobName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for job %s in cluster %s, got error: %v", jobName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func waitForJobOrFail(c *fedclientset.Clientset, namespace string, jobName string, clusters fedframework.ClusterSlice) {
	err := waitForJob(c, namespace, jobName, clusters)
	framework.ExpectNoError(err, "Failed to verify job %q/%q, err: %v", namespace, jobName, err)
}

func waitForJob(c *fedclientset.Clientset, namespace string, jobName string, clusters fedframework.ClusterSlice) error {
	err := wait.Poll(10*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		fjob, err := c.Batch().Jobs(namespace).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		succeeded := int32(0)
		for _, cluster := range clusters {
			job, err := cluster.Batch().Jobs(namespace).Get(jobName, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				By(fmt.Sprintf("Failed getting job: %q/%q/%q, err: %v", cluster.Name, namespace, jobName, err))
				return false, err
			}
			if err == nil {
				if !verifyJob(fjob, job) {
					By(fmt.Sprintf("Job meta or spec not match for cluster %q:\n    federation: %v\n    cluster: %v", cluster.Name, fjob, job))
					return false, nil
				}
				succeeded += job.Status.Succeeded
			}
		}
		if succeeded == fjob.Status.Succeeded &&
			(fjob.Spec.Completions != nil && succeeded == *fjob.Spec.Completions) {
			return true, nil
		}
		By(fmt.Sprintf("Job statuses not match, federation succeeded: %v/%v, clusters succeeded: %v\n",
			fjob.Status.Succeeded, func(p *int32) int32 {
				if p != nil {
					return *p
				} else {
					return -1
				}
			}(fjob.Spec.Completions), succeeded))
		return false, nil
	})

	return err
}

func verifyJob(fedJob, localJob *batchv1.Job) bool {
	localJobObj, _ := api.Scheme.DeepCopy(localJob)
	localJob = localJobObj.(*batchv1.Job)
	localJob.Spec.ManualSelector = fedJob.Spec.ManualSelector
	localJob.Spec.Completions = fedJob.Spec.Completions
	localJob.Spec.Parallelism = fedJob.Spec.Parallelism
	localJob.Spec.BackoffLimit = fedJob.Spec.BackoffLimit
	return fedutil.ObjectMetaAndSpecEquivalent(fedJob, localJob)
}

func createJobOrFail(clientset *fedclientset.Clientset, namespace string) *batchv1.Job {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createJobOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation job %q in namespace %q", FederationJobName, namespace))

	job := newJobForFed(namespace, FederationJobName, 5, 5)

	_, err := clientset.Batch().Jobs(namespace).Create(job)
	framework.ExpectNoError(err, "Creating job %q in namespace %q", job.Name, namespace)
	By(fmt.Sprintf("Successfully created federation job %q in namespace %q", FederationJobName, namespace))
	return job
}

func deleteJobOrFail(clientset *fedclientset.Clientset, nsName string, jobName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting job %q in namespace %q", jobName, nsName))
	err := clientset.Batch().Jobs(nsName).Delete(jobName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting job %q in namespace %q", jobName, nsName)
	}

	// Wait for the job to be deleted.
	err = wait.Poll(10*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		_, err := clientset.Batch().Jobs(nsName).Get(jobName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting job %s: %v", jobName, err)
	}
}

func newJobForFed(namespace string, name string, completions int32, parallelism int32) *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: batchv1.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": "fjob"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "sleep",
							Image:   imageutils.GetBusyBoxImage(),
							Command: []string{"sleep", "1"},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			},
		},
	}
}
