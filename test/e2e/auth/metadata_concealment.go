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

package auth

import (
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	imageutil "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("Metadata Concealment", func() {
	f := framework.NewDefaultFramework("metadata-concealment")

	ginkgo.It("should run a check-metadata-concealment job to completion", func() {
		e2eskipper.SkipUnlessProviderIs("gce")
		ginkgo.By("Creating a job")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name: "check-metadata-concealment",
			},
			Spec: batchv1.JobSpec{
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Name: "check-metadata-concealment",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "check-metadata-concealment",
								Image: imageutil.GetE2EImage(imageutil.CheckMetadataConcealment),
							},
						},
						RestartPolicy: v1.RestartPolicyOnFailure,
					},
				},
			},
		}
		job, err := e2ejob.CreateJob(f.ClientSet, f.Namespace.Name, job)
		framework.ExpectNoError(err, "failed to create job (%s:%s)", f.Namespace.Name, job.Name)

		ginkgo.By("Ensuring job reaches completions")
		err = e2ejob.WaitForJobComplete(f.ClientSet, f.Namespace.Name, job.Name, int32(1))
		framework.ExpectNoError(err, "failed to ensure job completion (%s:%s)", f.Namespace.Name, job.Name)
	})
})
