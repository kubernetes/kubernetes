/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	"context"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Pull Image [Serial]", func() {
	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)
	nginxNewImage := imageutils.GetE2EImage(imageutils.NginxNew)
	nginxPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx",
				Image:           nginxImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	nginxNewPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx-new",
				Image:           nginxNewImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	ginkgo.Context("serialize image pull", func() {
		ginkgo.It("should be waiting more", func(ctx context.Context) {
			framework.Logf("Creating pod %q", nginxPodDesc.Name)
			framework.Logf("Creating pod %q", nginxNewPodDesc.Name)
		})
	})
})
