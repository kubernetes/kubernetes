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

package upgrades

import (
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// JobUpgradeTest tests that a Job is running before, during and after
// a cluster upgrade.
type JobUpgradeTest struct {
	job *batch.Job
}

// Setup creates a Job and then verifies that it's running
func (t *JobUpgradeTest) Setup(f *framework.Framework) {
	jobName := "job-upgradetest"
	namespaceName := "job-upgrade"

	// Grab a unique namespace so we don't collide.
	ns, err := f.CreateNamespace(namespaceName, nil)
	framework.ExpectNoError(err)

	t.job = newTestJob("notTerminate", jobName, v1.RestartPolicyNever, 1, 1)

	By("Creating a Job")
	if t.job, err = f.ClientSet.Batch().Jobs(ns.Name).Create(t.job); err != nil {
		framework.Failf("unable to create test Job %s: %v", t.job.Name, err)
	}

	By("Waiting for Job pods to be running")
	waitForAllPodsRunning(f, t.job.Namespace, t.job.Name, 1)

	By("Validating the Job after creation")
	t.validateRunningJob(f)
}

// Test waits for the upgrade to complete, and then verifies that
// the Job is still running
func (t *JobUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	testDuringDisruption := upgrade == MasterUpgrade

	if testDuringDisruption {
		By("validating the Job is still running during upgrade")
		wait.Until(func() {
			t.validateRunningJob(f)
		}, framework.Poll, done)
	}

	<-done

	By("validating the Job is still running after upgrade")
	t.validateRunningJob(f)
}

// Teardown cleans up any remaining resources.
func (t *JobUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func (t *JobUpgradeTest) validateRunningJob(f *framework.Framework) {
	By("confirming the Job pods are all running")
	res, err := checkForAllPodsRunning(f, t.job.Namespace, t.job.Name, *t.job.Spec.Parallelism)
	framework.ExpectNoError(err)

	if !res {
		framework.Failf("expected all job pods to be running, but they weren't")
	}

	// Job resource itself should be good
	By("confirming the Job resource is in a good state")
	err = checkJobStatus(f, t.job.Namespace, t.job.Name)
	framework.ExpectNoError(err)
}

// newTestJob returns a job which does one of several testing behaviors.
func newTestJob(behavior, name string, rPol v1.RestartPolicy, parallelism, completions int32) *batch.Job {
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: batch.JobSpec{
			Parallelism:    &parallelism,
			Completions:    &completions,
			ManualSelector: newBool(false),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"job": name},
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
							Name:    name,
							Image:   "gcr.io/google_containers/busybox:1.24",
							Command: []string{"sleep", "1000000"},
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

	return job
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}

func checkJobStatus(f *framework.Framework, namespace string, jobName string) error {
	job, err := f.ClientSet.Batch().Jobs(namespace).Get(jobName, metav1.GetOptions{})

	if err != nil {
		return fmt.Errorf("could not get job from v1")
	}

	if job.Status.Succeeded != 0 {
		return fmt.Errorf("Expected zero succeeded pods, got %v", job.Status.Succeeded)
	} else if job.Status.Failed != 0 {
		return fmt.Errorf("Expected zero failed pods, got %v", job.Status.Failed)
	} else if job.Status.Active != 1 {
		return fmt.Errorf("Expected one active pod, got %v", job.Status.Active)
	}

	return nil
}

// Wait for all pods to become Running.  Only use when pods will run for a long time, or it will be racy.
func waitForAllPodsRunning(f *framework.Framework, ns, jobName string, parallelism int32) error {
	return wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		return checkForAllPodsRunning(f, ns, jobName, parallelism)
	})
}

func checkForAllPodsRunning(f *framework.Framework, ns, jobName string, parallelism int32) (bool, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"job": jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := f.ClientSet.Core().Pods(ns).List(options)
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
