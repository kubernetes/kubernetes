//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Shortened Grace Period", func() {
	f := framework.NewDefaultFramework("shortened-grace-period")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	ginkgo.Context("When repeatedly deleting pods", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("should be deleted immediately", func() {
			const (
				gracePeriod      = 10
				gracePeriodShort = 1
			)
			podName := "test"
			podClient.CreateSync(context.TODO(), getGracePeriodTestPod(podName, gracePeriod))
			err := podClient.Delete(context.TODO(), podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err)
			start := time.Now()
			podClient.DeleteSync(context.TODO(), podName, *metav1.NewDeleteOptions(gracePeriodShort), gracePeriod*time.Second)
			gomega.Expect(time.Since(start)).To(gomega.BeNumerically("<", gracePeriod*time.Second))
		})
	})
})

func getGracePeriodTestPod(name string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "999999"},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	return pod
}
