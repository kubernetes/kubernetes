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

package storage

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Downward API", framework.WithSerial(), framework.WithDisruptive(), feature.EphemeralStorage, func() {
	f := framework.NewDefaultFramework("downward-api")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Downward API tests for local ephemeral storage", func() {
		ginkgo.It("should provide container's limits.ephemeral-storage and requests.ephemeral-storage as env vars", func(ctx context.Context) {
			podName := "downward-api-" + string(uuid.NewUUID())
			env := []v1.EnvVar{
				{
					Name: "EPHEMERAL_STORAGE_LIMIT",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "limits.ephemeral-storage",
						},
					},
				},
				{
					Name: "EPHEMERAL_STORAGE_REQUEST",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "requests.ephemeral-storage",
						},
					},
				},
			}
			expectations := []string{
				fmt.Sprintf("EPHEMERAL_STORAGE_LIMIT=%d", 64*1024*1024),
				fmt.Sprintf("EPHEMERAL_STORAGE_REQUEST=%d", 32*1024*1024),
			}

			testDownwardAPIForEphemeralStorage(ctx, f, podName, env, expectations)
		})

		ginkgo.It("should provide default limits.ephemeral-storage from node allocatable", func(ctx context.Context) {
			podName := "downward-api-" + string(uuid.NewUUID())
			env := []v1.EnvVar{
				{
					Name: "EPHEMERAL_STORAGE_LIMIT",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "limits.ephemeral-storage",
						},
					},
				},
			}
			expectations := []string{
				"EPHEMERAL_STORAGE_LIMIT=[1-9]",
			}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podName,
					Labels: map[string]string{"name": podName},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "dapi-container",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c", "env"},
							Env:     env,
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
		})
	})

})

func testDownwardAPIForEphemeralStorage(ctx context.Context, f *framework.Framework, podName string, env []v1.EnvVar, expectations []string) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "dapi-container",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "env"},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceEphemeralStorage: resource.MustParse("32Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceEphemeralStorage: resource.MustParse("64Mi"),
						},
					},
					Env: env,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
}

func testDownwardAPIUsingPod(ctx context.Context, f *framework.Framework, pod *v1.Pod, env []v1.EnvVar, expectations []string) {
	e2epodoutput.TestContainerOutputRegexp(ctx, f, "downward api env vars", pod, 0, expectations)
}
