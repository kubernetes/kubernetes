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

package podcheckpoint

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

// startAPIServer starts an in-process apiserver for the test. When serveCheckpointAPI
// is true it enables the alpha node.k8s.io/v1alpha1 group via --runtime-config;
// the PodLevelCheckpointRestore feature gate must be toggled separately by the caller
// (the gate also guards whether the group is served, see runtime_class.go).
func startAPIServer(tCtx ktesting.TContext, t *testing.T, serveCheckpointAPI bool, configure ...func(*options.ServerRunOptions)) (clientset.Interface, *restclient.Config, framework.TearDownFunc) {
	return framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable admission plugins that interfere with a minimal pod/node setup.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
			if serveCheckpointAPI {
				opts.APIEnablement.RuntimeConfig = cliflag.ConfigurationMap{
					nodev1alpha1.SchemeGroupVersion.String(): "true",
				}
			}
			for _, f := range configure {
				f(opts)
			}
		},
	})
}

func createCheckpointSource(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, namespace, name string) *v1.Pod {
	t.Helper()
	pod, err := client.CoreV1().Pods(namespace).Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return markCheckpointSourceRunning(tCtx, t, client, pod)
}

func bindCheckpointSource(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, pod *v1.Pod) *v1.Pod {
	t.Helper()
	pods := client.CoreV1().Pods(pod.Namespace)
	if err := pods.Bind(tCtx, &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: pod.Name, UID: pod.UID}, Target: v1.ObjectReference{Kind: "Node", Name: "node-1"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	bound, err := pods.Get(tCtx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return bound
}

func markCheckpointSourceRunning(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, pod *v1.Pod) *v1.Pod {
	t.Helper()
	var err error
	if pod.Spec.NodeName == "" {
		pod = bindCheckpointSource(tCtx, t, client, pod)
	}

	pod.Status.Phase = v1.PodRunning
	pod, err = client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return pod
}

func newPodCheckpoint(namespace, name, sourcePodName string) *nodev1alpha1.PodCheckpoint {
	return &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: nodev1alpha1.PodCheckpointSpec{
			SourcePod: nodev1alpha1.PodReference{Name: sourcePodName},
		},
	}
}
