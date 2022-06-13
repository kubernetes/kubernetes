/*
Copyright 2022 The Kubernetes Authors.

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

package windows

import (
	"context"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("StatefulSets", func() {
	f := framework.NewDefaultFramework("statefulsets")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var ns string
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ssName := "windows-ss"
	labels := map[string]string{
		"app": "agnhost",
	}
	headlessSvcName := "svc"

	ginkgo.It("should have the ability to delete and recreate pods for StatefulSets which preserve their ability to serve as routed endpoints for services", func() {

		ginkgo.By("Creating service " + headlessSvcName + " in namespace " + ns)
		headlessService := e2eservice.CreateServiceSpec(headlessSvcName, "", true, labels)
		_, err := c.CoreV1().Services(ns).Create(context.TODO(), headlessService, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a statefulset")
		ss := newStatefulSet(ssName, ns, headlessSvcName, 3, labels)
		_, err = c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)

		ginkgo.By("getting that endpoints of the statefulset")
		pods1 := e2estatefulset.GetPodList(c, ss)
		gomega.Expect(pods1.Items).To(gomega.HaveLen(int(*ss.Spec.Replicas)))

		ginkgo.By("deleting one of the endpoints of the statefulset")
		err = c.CoreV1().Pods(ns).Delete(context.TODO(), ssName+"-1", metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		time.Sleep(10 * time.Second)
		err = e2epod.WaitForPodNameRunningInNamespace(c, ssName+"-1", ns)
		framework.ExpectNoError(err)

		ginkgo.By("checking the length of the endpoints of the statefulset")
		pods2 := e2estatefulset.GetPodList(c, ss)
		gomega.Expect(pods2.Items).To(gomega.HaveLen(int(*ss.Spec.Replicas)))

	})
})

func newStatefulSet(name, ns, governingSvcName string, replicas int32, labels map[string]string) *appsv1.StatefulSet {

	return &appsv1.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{
						"kubernetes.io/os": "windows",
					},
					Containers: []v1.Container{
						{
							Name:  "agnhost",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
						},
					},
				},
			},
			ServiceName: governingSvcName,
		},
	}
}
