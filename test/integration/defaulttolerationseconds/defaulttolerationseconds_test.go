/*
Copyright 2017 The Kubernetes Authors.

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

package defaulttolerationseconds

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestAdmission(t *testing.T) {
	tCtx := ktesting.Init(t)
	handler, err := newHandlerForTest()
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerConfig: func(cfg *controlplane.Config) {
			cfg.ControlPlane.Generic.EnableProfiling = true
			cfg.ControlPlane.Generic.AdmissionControl = handler
		},
	})
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "default-toleration-seconds", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      "foo",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test",
					Image: "an-image",
				},
			},
		},
	}

	updatedPod, err := client.CoreV1().Pods(pod.Namespace).Create(tCtx, &pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating pod: %v", err)
	}

	var defaultSeconds int64 = 300
	nodeNotReady := v1.Toleration{
		Key:               v1.TaintNodeNotReady,
		Operator:          v1.TolerationOpExists,
		Effect:            v1.TaintEffectNoExecute,
		TolerationSeconds: &defaultSeconds,
	}

	nodeUnreachable := v1.Toleration{
		Key:               v1.TaintNodeUnreachable,
		Operator:          v1.TolerationOpExists,
		Effect:            v1.TaintEffectNoExecute,
		TolerationSeconds: &defaultSeconds,
	}

	found := 0
	tolerations := updatedPod.Spec.Tolerations
	for i := range tolerations {
		if found == 2 {
			break
		}
		if tolerations[i].MatchToleration(&nodeNotReady) {
			if helper.Semantic.DeepEqual(tolerations[i], nodeNotReady) {
				found++
				continue
			}
		}
		if tolerations[i].MatchToleration(&nodeUnreachable) {
			if helper.Semantic.DeepEqual(tolerations[i], nodeUnreachable) {
				found++
				continue
			}
		}
	}

	if found != 2 {
		t.Fatalf("unexpected tolerations: %v\n", updatedPod.Spec.Tolerations)
	}
}

// newHandlerForTest returns a handler configured for testing.
func newHandlerForTest() (*defaulttolerationseconds.Plugin, error) {
	handler := defaulttolerationseconds.NewDefaultTolerationSeconds()
	pluginInitializer := initializer.New(nil, nil, nil, nil, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	return handler, admission.ValidateInitialization(handler)
}
