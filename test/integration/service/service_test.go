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

package service

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration/framework"
)

// Test_ExternalNameServiceStopsDefaultingInternalTrafficPolicy tests that Services no longer default
// the internalTrafficPolicy field when Type is ExternalName. This test exists due to historic reasons where
// the internalTrafficPolicy field was being defaulted in older versions. New versions stop defauting the
// field and drop on read, but for compatibility reasons we still accept the field.
func Test_ExternalNameServiceStopsDefaultingInternalTrafficPolicy(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateTestingNamespace("test-external-name-drops-internal-traffic-policy", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: "foo.bar.com",
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}
}

// Test_ExternalNameServiceDropsInternalTrafficPolicy tests that Services accepts the internalTrafficPolicy field on Create,
// but drops the field on read. This test exists due to historic reasons where the internalTrafficPolicy field was being defaulted
// in older versions. New versions stop defauting the field and drop on read, but for compatibility reasons we still accept the field.
func Test_ExternalNameServiceDropsInternalTrafficPolicy(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateTestingNamespace("test-external-name-drops-internal-traffic-policy", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	internalTrafficPolicy := corev1.ServiceInternalTrafficPolicyCluster
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                  corev1.ServiceTypeExternalName,
			ExternalName:          "foo.bar.com",
			InternalTrafficPolicy: &internalTrafficPolicy,
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}
}
