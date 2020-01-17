/*
Copyright 2018 The Kubernetes Authors.

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

package node

import (
	"time"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/test/e2e/framework"
	jobutil "k8s.io/kubernetes/test/e2e/framework/job"

	"github.com/onsi/ginkgo"
)

const dummyFinalizer = "k8s.io/dummy-finalizer"

var _ = framework.KubeDescribe("[Feature:TTLAfterFinished][NodeAlphaFeature:TTLAfterFinished]", func() {
	f := framework.NewDefaultFramework("ttlafterfinished")

	ginkgo.It("job should be deleted once it finishes after TTL seconds", func() {
		testFinishedJob(f)
	})
})

func cleanupJob(f *framework.Framework, job *batchv1.Job) {
	ns := f.Namespace.Name
	c := f.ClientSet

	framework.Logf("Remove the Job's dummy finalizer; the Job should be deleted cascadingly")
	removeFinalizerFunc := func(j *batchv1.Job) {
		j.ObjectMeta.Finalizers = slice.RemoveString(j.ObjectMeta.Finalizers, dummyFinalizer, nil)
	}
	_, err := jobutil.UpdateJobWithRetries(c, ns, job.Name, removeFinalizerFunc)
	framework.ExpectNoError(err)
	jobutil.WaitForJobGone(c, ns, job.Name, wait.ForeverTestTimeout)

	err = jobutil.WaitForAllJobPodsGone(c, ns, job.Name)
	framework.ExpectNoError(err)
}

func testFinishedJob(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	parallelism := int32(1)
	completions := int32(1)
	backoffLimit := int32(2)
	ttl := int32(10)

	job := jobutil.NewTestJob("randomlySucceedOrFail", "rand-non-local", v1.RestartPolicyNever, parallelism, completions, nil, backoffLimit)
	job.Spec.TTLSecondsAfterFinished = &ttl
	job.ObjectMeta.Finalizers = []string{dummyFinalizer}
	defer cleanupJob(f, job)

	framework.Logf("Create a Job %s/%s with TTL", ns, job.Name)
	job, err := jobutil.CreateJob(c, ns, job)
	framework.ExpectNoError(err)

	framework.Logf("Wait for the Job to finish")
	err = jobutil.WaitForJobFinish(c, ns, job.Name)
	framework.ExpectNoError(err)

	framework.Logf("Wait for TTL after finished controller to delete the Job")
	err = jobutil.WaitForJobDeleting(c, ns, job.Name)
	framework.ExpectNoError(err)

	framework.Logf("Check Job's deletionTimestamp and compare with the time when the Job finished")
	job, err = jobutil.GetJob(c, ns, job.Name)
	framework.ExpectNoError(err)
	finishTime := jobutil.FinishTime(job)
	finishTimeUTC := finishTime.UTC()
	framework.ExpectNotEqual(finishTime.IsZero(), true)

	deleteAtUTC := job.ObjectMeta.DeletionTimestamp.UTC()
	framework.ExpectNotEqual(deleteAtUTC, nil)

	expireAtUTC := finishTimeUTC.Add(time.Duration(ttl) * time.Second)
	framework.ExpectEqual(deleteAtUTC.Before(expireAtUTC), false)
}
