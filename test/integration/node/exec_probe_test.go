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

package node

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestExtendedExecProbeTimeout(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExtendDefaultExecProbeTimeout, true)()

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

	_, err = client.CoreV1().ServiceAccounts(ns.Name).Create(context.TODO(), &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: ns.Name},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "with-exec-probe-timeout",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "with-probe-timeout",
					Image: "some:image",
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{
								Command: []string{
									"ls",
								},
							},
						},
					},
				},
			},
		},
	}

	if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("failed to create pod: %v", err)
	}

	pod, err = client.CoreV1().Pods(ns.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("error getting pod: %v", err)
	}

	if pod.Spec.Containers[0].LivenessProbe.TimeoutSeconds != 5 {
		t.Errorf("unexpected timeout for exec probe, got %d but expected 5", pod.Spec.Containers[0].LivenessProbe.TimeoutSeconds)
	}
}
