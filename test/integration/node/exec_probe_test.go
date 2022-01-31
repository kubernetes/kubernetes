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
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestExtendedExecProbeTimeout(t *testing.T) {
	testServer := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--enable-admission-plugins", "ExtendDefaultExecProbeTimeout",
	}, framework.SharedEtcd())
	t.Cleanup(testServer.TearDownFn)

	client := clientset.NewForConfigOrDie(testServer.ClientConfig)

	testNamespace := "extended-exec-probe-timeout"
	ns, err := client.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.CoreV1().ServiceAccounts(testNamespace).Create(context.TODO(), &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: testNamespace},
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
