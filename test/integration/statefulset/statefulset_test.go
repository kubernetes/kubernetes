/*
Copyright 2015 The Kubernetes Authors.

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

package statefulset

import (
	"testing"

	//metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestStatefulSetScaleSubresource(t *testing.T) {
	s, closeFn, sc, informers, c := scSetup(t)
	defer closeFn()
	name := "test-new-statefulset"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int(1)
	tester := &statefulsetTester{t: t, c: c, service: newService(ns.Name), statefulset: newStatefulset(name, ns.Name, replicas)}

	ss, err := c.Apps().StatefulSets(ns.Name).Create(tester.statefulset)
	if err != nil {
		t.Fatalf("failed to create statefulset %s: %v", ss.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go sc.Run(5, stopCh)

	// Wait for the StatefulSet to be updated to revision 1
	err = tester.waitForStatefulSetImage("1", fakeImage)
	if err != nil {
		t.Fatalf("Failed to wait for statefulset %s's image: %v", ss.Name, err)
	}

	// Make sure the StatefulSet status becomes valid while manually marking StatefulSet pods as ready at the same time
	tester.waitForStatefulSetStatusValidAndMarkPodsReady()

	// Getting scale subresource
	scale, err := c.Apps().Scales(ns.Name).Get("Scale", ss.Name)
	if err != nil {
		t.Fatalf("Failed to get scale subresource: %v", err)
	}
	if scale.Spec.Replicas != int32(1) {
		t.Errorf("Spec.Replicas of StatefulSet %s's scale subresource should be 1, got: %d", ss.Name, scale.Spec.Replicas)
	}
	if scale.Status.Replicas != int32(1) {
		t.Errorf("Status.Replicas of StatefulSet %s's scale subresource should be 1, got: %d", ss.Name, scale.Status.Replicas)
	}

	// Updating the scale subresource by changing Spec.Replicas from 1 to 2
	scale.ResourceVersion = "" //unconditionally update to 2 replicas
	scale.Spec.Replicas = int32(2)

	// Verifying Spec.Replicas was modified
	scale, err = c.Apps().Scales(ns.Name).Get("Scale", ss.Name)
	if err != nil {
		t.Fatalf("Failed to get scale subresource: %v", err)
	}
	if scale.Spec.Replicas != int32(2) {
		t.Errorf("Spec.Replicas of StatefulSet %s's scale subresource should be 2, got: %d", ss.Name, scale.Spec.Replicas)
	}
}
