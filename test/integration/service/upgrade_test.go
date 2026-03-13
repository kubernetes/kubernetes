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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/test/integration/framework"
)

func Test_UpgradeService(t *testing.T) {
	etcdOptions := framework.SharedEtcd()
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), etcdOptions)
	defer s.TearDownFn()
	serviceName := "test-old-service"
	ns := "old-service-ns"

	kubeclient, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if _, err := kubeclient.CoreV1().Namespaces().Create(context.TODO(), (&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}), metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// Create a service and store it in etcd with missing fields representing an old version
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         ns,
			CreationTimestamp: metav1.Now(),
			UID:               "08675309-9376-9376-9376-086753099999",
		},
		Spec: v1.ServiceSpec{
			ClusterIP: "10.0.0.1",
			Ports: []v1.ServicePort{
				{
					Name: "test-port",
					Port: 81,
				},
			},
		},
	}
	svcJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), svc)
	if err != nil {
		t.Fatalf("Failed creating service JSON: %v", err)
	}
	key := "/" + etcdOptions.Prefix + "/services/specs/" + ns + "/" + serviceName
	if _, err := s.EtcdClient.Put(context.Background(), key, string(svcJSON)); err != nil {
		t.Error(err)
	}
	t.Logf("Service stored in etcd %v", string(svcJSON))

	// Try to update the service
	_, err = kubeclient.CoreV1().Services(ns).Update(context.TODO(), svc, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}
}
