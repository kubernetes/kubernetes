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

package storage

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	volumePath = "/test-volume"
	volumeName = "test-volume"
	mountImage = imageutils.GetE2EImage(imageutils.Mounttest)
)

var _ = utils.SIGDescribe("Ephemeralstorage", func() {
	var (
		c clientset.Interface
	)

	f := framework.NewDefaultFramework("pv")

	BeforeEach(func() {
		c = f.ClientSet
	})

	Describe("When pod refers to non-existent ephemeral storage", func() {
		for _, testSource := range invalidEphemeralSource("pod-ephm-test") {
			It(fmt.Sprintf("should allow deletion of pod with invalid volume : %s", testSource.volumeType), func() {
				pod := testEphemeralVolumePod(f, testSource.volumeType, testSource.source)
				pod, err := c.CoreV1().Pods(f.Namespace.Name).Create(pod)
				Expect(err).NotTo(HaveOccurred())

				// Allow it to sleep for 30 seconds
				time.Sleep(30 * time.Second)
				framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
				framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
			})
		}
	})
})

type ephemeralTestInfo struct {
	volumeType string
	source     *v1.VolumeSource
}

func testEphemeralVolumePod(f *framework.Framework, volumeType string, source *v1.VolumeSource) *v1.Pod {
	var (
		suffix = strings.ToLower(fmt.Sprintf("%s-%s", volumeType, rand.String(4)))
	)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("pod-ephm-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  fmt.Sprintf("test-container-subpath-%s", suffix),
					Image: mountImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         volumeName,
					VolumeSource: *source,
				},
			},
		},
	}
}

func invalidEphemeralSource(suffix string) []ephemeralTestInfo {
	testInfo := []ephemeralTestInfo{
		{
			volumeType: "secret",
			source: &v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: fmt.Sprintf("secert-%s", suffix),
				},
			},
		},
		{
			volumeType: "configmap",
			source: &v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: fmt.Sprintf("configmap-%s", suffix),
					},
				},
			},
		},
		{
			volumeType: "projected",
			source: &v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{
					Sources: []v1.VolumeProjection{
						{
							Secret: &v1.SecretProjection{
								LocalObjectReference: v1.LocalObjectReference{
									Name: fmt.Sprintf("secret-%s", suffix),
								},
							},
						},
					},
				},
			},
		},
	}
	return testInfo
}
