/*
Copyright 2020 The Kubernetes Authors.

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

package network

import (
	"context"
	"strings"
	"time"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("IngressClass [Feature:Ingress]", func() {
	f := framework.NewDefaultFramework("ingressclass")
	var cs clientset.Interface
	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.It("should set default value on new IngressClass", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", true)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingress, err := createBasicIngress(cs, f.Namespace.Name)
		framework.ExpectNoError(err)

		if ingress.Spec.IngressClassName == nil {
			framework.Failf("Expected IngressClassName to be set by Admission Controller")
		} else if *ingress.Spec.IngressClassName != ingressClass1.Name {
			framework.Failf("Expected IngressClassName to be %s, got %s", ingressClass1.Name, *ingress.Spec.IngressClassName)
		}
	})

	ginkgo.It("should not set default value if no default IngressClass", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", false)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingress, err := createBasicIngress(cs, f.Namespace.Name)
		framework.ExpectNoError(err)

		if ingress.Spec.IngressClassName != nil {
			framework.Failf("Expected IngressClassName to be nil, got %s", *ingress.Spec.IngressClassName)
		}
	})

	ginkgo.It("should prevent Ingress creation if more than 1 IngressClass marked as default", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", true)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingressClass2, err := createIngressClass(cs, "ingressclass2", true)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass2.Name)

		// the admission controller may take a few seconds to observe both ingress classes
		expectedErr := "2 default IngressClasses were found, only 1 allowed"
		var lastErr error
		if err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
			defer cs.NetworkingV1beta1().Ingresses(f.Namespace.Name).Delete(context.TODO(), "ingress1", metav1.DeleteOptions{})
			_, err := createBasicIngress(cs, f.Namespace.Name)
			if err == nil {
				return false, nil
			}
			lastErr = err
			return strings.Contains(err.Error(), expectedErr), nil
		}); err != nil {
			framework.Failf("Expected error to contain %s, got %s", expectedErr, lastErr.Error())
		}
	})

})

func createIngressClass(cs clientset.Interface, name string, isDefault bool) (*networkingv1beta1.IngressClass, error) {
	ingressClass := &networkingv1beta1.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1beta1.IngressClassSpec{
			Controller: "example.com/controller",
		},
	}

	if isDefault {
		ingressClass.Annotations = map[string]string{networkingv1beta1.AnnotationIsDefaultIngressClass: "true"}
	}

	return cs.NetworkingV1beta1().IngressClasses().Create(context.TODO(), ingressClass, metav1.CreateOptions{})
}

func createBasicIngress(cs clientset.Interface, namespace string) (*networkingv1beta1.Ingress, error) {
	return cs.NetworkingV1beta1().Ingresses(namespace).Create(context.TODO(), &networkingv1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: "ingress1",
		},
		Spec: networkingv1beta1.IngressSpec{
			Backend: &networkingv1beta1.IngressBackend{
				ServiceName: "default-backend",
				ServicePort: intstr.FromInt(80),
			},
		},
	}, metav1.CreateOptions{})
}

func deleteIngressClass(cs clientset.Interface, name string) {
	err := cs.NetworkingV1beta1().IngressClasses().Delete(context.TODO(), name, metav1.DeleteOptions{})
	framework.ExpectNoError(err)
}
