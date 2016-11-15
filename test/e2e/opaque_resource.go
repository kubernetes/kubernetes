/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Opaque resources [Feature:OpaqueResources]", func() {
	f := framework.NewDefaultFramework("opaque-resource")
	opaqueResName := api.OpaqueIntResourceName("foo")
	var node *api.Node

	BeforeEach(func() {
		if node == nil {
			// Priming invocation; select the first non-master node.
			nodes, err := f.ClientSet.Core().Nodes().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, n := range nodes.Items {
				if !system.IsMasterNode(&n) {
					node = &n
					break
				}
			}
			if node == nil {
				Fail("unable to select a non-master node")
			}
		}

		removeOpaqueResource(f, node.Name, opaqueResName)
		addOpaqueResource(f, node.Name, opaqueResName)
	})

	It("should not break pods that do not consume opaque integer resources.", func() {
		By("Creating a vanilla pod")
		requests := api.ResourceList{api.ResourceCPU: resource.MustParse("0.1")}
		limits := api.ResourceList{api.ResourceCPU: resource.MustParse("0.2")}
		pod := newTestPod(f, "without-oir", requests, limits)

		By("Observing an event that indicates the pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		predicate := func(e *api.Event) bool {
			return e.Type == api.EventTypeNormal &&
				e.Reason == "Scheduled" &&
				// Here we don't check for the bound node name since it can land on
				// any one (this pod doesn't require any of the opaque resource.)
				strings.Contains(e.Message, fmt.Sprintf("Successfully assigned %v", pod.Name))
		}
		success, err := observeEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should schedule pods that do consume opaque integer resources.", func() {
		By("Creating a pod that requires less of the opaque resource than is allocatable on a node.")
		requests := api.ResourceList{
			api.ResourceCPU: resource.MustParse("0.1"),
			opaqueResName:   resource.MustParse("1"),
		}
		limits := api.ResourceList{
			api.ResourceCPU: resource.MustParse("0.2"),
			opaqueResName:   resource.MustParse("2"),
		}
		pod := newTestPod(f, "min-oir", requests, limits)

		By("Observing an event that indicates the pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		predicate := func(e *api.Event) bool {
			return e.Type == api.EventTypeNormal &&
				e.Reason == "Scheduled" &&
				strings.Contains(e.Message, fmt.Sprintf("Successfully assigned %v to %v", pod.Name, node.Name))
		}
		success, err := observeEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should not schedule pods that exceed the available amount of opaque integer resource.", func() {
		By("Creating a pod that requires more of the opaque resource than is allocatable on any node")
		requests := api.ResourceList{opaqueResName: resource.MustParse("6")}
		limits := api.ResourceList{}

		By("Observing an event that indicates the pod was not scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(newTestPod(f, "over-max-oir", requests, limits))
			return err
		}
		predicate := func(e *api.Event) bool {
			return e.Type == "Warning" &&
				e.Reason == "FailedScheduling" &&
				strings.Contains(e.Message, "failed to fit in any node")
		}
		success, err := observeEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should account opaque integer resources in pods with multiple containers.", func() {
		By("Creating a pod with two containers that together require less of the opaque resource than is allocatable on a node")
		requests := api.ResourceList{opaqueResName: resource.MustParse("1")}
		limits := api.ResourceList{}
		image := framework.GetPauseImageName(f.ClientSet)
		// This pod consumes 2 "foo" resources.
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "mult-container-oir",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "pause",
						Image: image,
						Resources: api.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
					{
						Name:  "pause-sidecar",
						Image: image,
						Resources: api.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
				},
			},
		}

		By("Observing an event that indicates the pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		predicate := func(e *api.Event) bool {
			return e.Type == api.EventTypeNormal &&
				e.Reason == "Scheduled" &&
				strings.Contains(e.Message, fmt.Sprintf("Successfully assigned %v to %v", pod.Name, node.Name))
		}
		success, err := observeEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))

		By("Creating a pod with two containers that together require more of the opaque resource than is allocatable on any node")
		requests = api.ResourceList{opaqueResName: resource.MustParse("3")}
		limits = api.ResourceList{}
		// This pod consumes 6 "foo" resources.
		pod = &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "mult-container-over-max-oir",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "pause",
						Image: image,
						Resources: api.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
					{
						Name:  "pause-sidecar",
						Image: image,
						Resources: api.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
				},
			},
		}

		By("Observing an event that indicates the pod was not scheduled")
		action = func() error {
			_, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		predicate = func(e *api.Event) bool {
			return e.Type == "Warning" &&
				e.Reason == "FailedScheduling" &&
				strings.Contains(e.Message, "failed to fit in any node")
		}
		success, err = observeEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})
})

// Adds the opaque resource to a node.
func addOpaqueResource(f *framework.Framework, nodeName string, opaqueResName api.ResourceName) {
	action := func() error {
		patch := []byte(fmt.Sprintf(`[{"op": "add", "path": "/status/capacity/%s", "value": "5"}]`, escapeForJSONPatch(opaqueResName)))
		return f.ClientSet.Core().RESTClient().Patch(api.JSONPatchType).Resource("nodes").Name(nodeName).SubResource("status").Body(patch).Do().Error()
	}
	predicate := func(n *api.Node) bool {
		capacity, foundCap := n.Status.Capacity[opaqueResName]
		allocatable, foundAlloc := n.Status.Allocatable[opaqueResName]
		return foundCap && capacity.MilliValue() == int64(5000) &&
			foundAlloc && allocatable.MilliValue() == int64(5000)
	}
	success, err := observeNodeUpdateAfterAction(f, nodeName, predicate, action)
	Expect(err).NotTo(HaveOccurred())
	Expect(success).To(Equal(true))
}

// Removes the opaque resource from a node.
func removeOpaqueResource(f *framework.Framework, nodeName string, opaqueResName api.ResourceName) {
	action := func() error {
		patch := []byte(fmt.Sprintf(`[{"op": "remove", "path": "/status/capacity/%s"}]`, escapeForJSONPatch(opaqueResName)))
		f.ClientSet.Core().RESTClient().Patch(api.JSONPatchType).Resource("nodes").Name(nodeName).SubResource("status").Body(patch).Do()
		return nil // Ignore error -- the opaque resource may not exist.
	}
	predicate := func(n *api.Node) bool {
		_, foundCap := n.Status.Capacity[opaqueResName]
		_, foundAlloc := n.Status.Allocatable[opaqueResName]
		return !foundCap && !foundAlloc
	}
	success, err := observeNodeUpdateAfterAction(f, nodeName, predicate, action)
	Expect(err).NotTo(HaveOccurred())
	Expect(success).To(Equal(true))
}

func escapeForJSONPatch(resName api.ResourceName) string {
	// Escape forward slashes in the resource name per the JSON Pointer spec.
	// See https://tools.ietf.org/html/rfc6901#section-3
	return strings.Replace(string(resName), "/", "~1", -1)
}

// Returns true if a node update matching the predicate was emitted from the
// system after performing the supplied action.
func observeNodeUpdateAfterAction(f *framework.Framework, nodeName string, nodePredicate func(*api.Node) bool, action func() error) (bool, error) {
	observedMatchingNode := false
	nodeSelector := fields.OneTermEqualSelector("metadata.name", nodeName)
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.FieldSelector = nodeSelector
				ls, err := f.ClientSet.Core().Nodes().List(options)
				return ls, err
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.FieldSelector = nodeSelector
				w, err := f.ClientSet.Core().Nodes().Watch(options)
				// Signal parent goroutine that watching has begun.
				informerStartedGuard.Do(func() { close(informerStartedChan) })
				return w, err
			},
		},
		&api.Node{},
		0,
		cache.ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				n, ok := newObj.(*api.Node)
				Expect(ok).To(Equal(true))
				if nodePredicate(n) {
					observedMatchingNode = true
				}
			},
		},
	)

	// Start the informer and block this goroutine waiting for the started signal.
	informerStopChan := make(chan struct{})
	defer func() { close(informerStopChan) }()
	go controller.Run(informerStopChan)
	<-informerStartedChan

	// Invoke the action function.
	err := action()
	if err != nil {
		return false, err
	}

	// Poll whether the informer has found a matching node update with a timeout.
	// Wait up 2 minutes polling every second.
	timeout := 2 * time.Minute
	interval := 1 * time.Second
	err = wait.Poll(interval, timeout, func() (bool, error) {
		return observedMatchingNode, nil
	})
	return err == nil, err
}

// Returns true if an event matching the predicate was emitted from the system
// after performing the supplied action.
func observeEventAfterAction(f *framework.Framework, eventPredicate func(*api.Event) bool, action func() error) (bool, error) {
	observedMatchingEvent := false

	// Create an informer to list/watch events from the test framework namespace.
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				ls, err := f.ClientSet.Core().Events(f.Namespace.Name).List(options)
				return ls, err
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				w, err := f.ClientSet.Core().Events(f.Namespace.Name).Watch(options)
				return w, err
			},
		},
		&api.Event{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				e, ok := obj.(*api.Event)
				By(fmt.Sprintf("Considering event: \nType = [%s], Reason = [%s], Message = [%s]", e.Type, e.Reason, e.Message))
				Expect(ok).To(Equal(true))
				if ok && eventPredicate(e) {
					observedMatchingEvent = true
				}
			},
		},
	)

	informerStopChan := make(chan struct{})
	defer func() { close(informerStopChan) }()
	go controller.Run(informerStopChan)

	// Invoke the action function.
	err := action()
	if err != nil {
		return false, err
	}

	// Poll whether the informer has found a matching event with a timeout.
	// Wait up 2 minutes polling every second.
	timeout := 2 * time.Minute
	interval := 1 * time.Second
	err = wait.Poll(interval, timeout, func() (bool, error) {
		return observedMatchingEvent, nil
	})
	return err == nil, err
}
