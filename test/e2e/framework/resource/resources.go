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

package resource

import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// Number of objects that gc can delete in a second.
	// GC issues 2 requestes for single delete.
	gcThroughput = 10
)

// ScaleResource scales resource to the given size.
func ScaleResource(
	clientset clientset.Interface,
	scalesGetter scaleclient.ScalesGetter,
	ns, name string,
	size uint,
	wait bool,
	kind schema.GroupKind,
	gvr schema.GroupVersionResource,
) error {
	ginkgo.By(fmt.Sprintf("Scaling %v %s in namespace %s to %d", kind, name, ns, size))
	if err := testutils.ScaleResourceWithRetries(scalesGetter, ns, name, size, gvr); err != nil {
		return fmt.Errorf("error while scaling RC %s to %d replicas: %v", name, size, err)
	}
	if !wait {
		return nil
	}
	return WaitForControlledPodsRunning(clientset, ns, name, kind)
}

// DeleteResourceAndWaitForGC deletes only given resource and waits for GC to delete the pods.
func DeleteResourceAndWaitForGC(c clientset.Interface, kind schema.GroupKind, ns, name string) error {
	ginkgo.By(fmt.Sprintf("deleting %v %s in namespace %s, will wait for the garbage collector to delete the pods", kind, name, ns))

	rtObject, err := GetRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			framework.Logf("%v %s not found: %v", kind, name, err)
			return nil
		}
		return err
	}
	selector, err := GetSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := GetReplicasFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}

	ps, err := testutils.NewPodStore(c, ns, selector, fields.Everything())
	if err != nil {
		return err
	}

	defer ps.Stop()
	falseVar := false
	deleteOption := metav1.DeleteOptions{OrphanDependents: &falseVar}
	startTime := time.Now()
	if err := testutils.DeleteResourceWithRetries(c, kind, ns, name, deleteOption); err != nil {
		return err
	}
	deleteTime := time.Since(startTime)
	framework.Logf("Deleting %v %s took: %v", kind, name, deleteTime)

	var interval, timeout time.Duration
	switch {
	case replicas < 100:
		interval = 100 * time.Millisecond
	case replicas < 1000:
		interval = 1 * time.Second
	default:
		interval = 10 * time.Second
	}
	if replicas < 5000 {
		timeout = 10 * time.Minute
	} else {
		timeout = time.Duration(replicas/gcThroughput) * time.Second
		// gcThroughput is pretty strict now, add a bit more to it
		timeout = timeout + 3*time.Minute
	}

	err = waitForPodsInactive(ps, interval, timeout)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Since(startTime) - deleteTime
	framework.Logf("Terminating %v %s pods took: %v", kind, name, terminatePodTime)

	// In gce, at any point, small percentage of nodes can disappear for
	// ~10 minutes due to hostError. 20 minutes should be long enough to
	// restart VM in that case and delete the pod.
	err = waitForPodsGone(ps, interval, 20*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

// waitForPodsGone waits until there are no pods left in the PodStore.
func waitForPodsGone(ps *testutils.PodStore, interval, timeout time.Duration) error {
	var pods []*v1.Pod
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		if pods = ps.List(); len(pods) == 0 {
			return true, nil
		}
		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		for _, pod := range pods {
			framework.Logf("ERROR: Pod %q still exists. Node: %q", pod.Name, pod.Spec.NodeName)
		}
		return fmt.Errorf("there are %d pods left. E.g. %q on node %q", len(pods), pods[0].Name, pods[0].Spec.NodeName)
	}
	return err
}

// waitForPodsInactive waits until there are no active pods left in the PodStore.
// This is to make a fair comparison of deletion time between DeleteRCAndPods
// and DeleteRCAndWaitForGC, because the RC controller decreases status.replicas
// when the pod is inactvie.
func waitForPodsInactive(ps *testutils.PodStore, interval, timeout time.Duration) error {
	var activePods []*v1.Pod
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods := ps.List()
		activePods = e2epod.FilterActivePods(pods)
		if len(activePods) != 0 {
			return false, nil
		}
		return true, nil
	})

	if err == wait.ErrWaitTimeout {
		for _, pod := range activePods {
			framework.Logf("ERROR: Pod %q running on %q is still active", pod.Name, pod.Spec.NodeName)
		}
		return fmt.Errorf("there are %d active pods. E.g. %q on node %q", len(activePods), activePods[0].Name, activePods[0].Spec.NodeName)
	}
	return err
}

// WaitForControlledPodsRunning waits up to 10 minutes for pods to become Running.
func WaitForControlledPodsRunning(c clientset.Interface, ns, name string, kind schema.GroupKind) error {
	rtObject, err := GetRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		return err
	}
	selector, err := GetSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := GetReplicasFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	err = testutils.WaitForEnoughPodsWithLabelRunning(c, ns, selector, int(replicas))
	if err != nil {
		return fmt.Errorf("Error while waiting for replication controller %s pods to be running: %v", name, err)
	}
	return nil
}

// WaitForControlledPods waits up to podListTimeout for getting pods of the specified controller name and return them.
func WaitForControlledPods(c clientset.Interface, ns, name string, kind schema.GroupKind) (pods *v1.PodList, err error) {
	rtObject, err := GetRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		return nil, err
	}
	selector, err := GetSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return nil, err
	}
	return e2epod.WaitForPodsWithLabel(c, ns, selector)
}
