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

package scheduling

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Opaque resources [Feature:OpaqueResources]", func() {
	f := framework.NewDefaultFramework("opaque-resource")
	opaqueResName := v1helper.OpaqueIntResourceName("foo")
	var node *v1.Node

	BeforeEach(func() {
		if node == nil {
			// Priming invocation; select the first non-master node.
			nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, n := range nodes.Items {
				if !system.IsMasterNode(n.Name) {
					node = &n
					break
				}
			}
			if node == nil {
				framework.Failf("unable to select a non-master node")
			}
		}

		addOpaqueResource(f, node.Name, opaqueResName)
	})

	// TODO: The suite times out if removeOpaqueResource is called as part of
	//       an AfterEach closure. For now, it is the last statement in each
	//       It block.
	// AfterEach(func() {
	// 	removeOpaqueResource(f, node.Name, opaqueResName)
	// })

	It("should not break pods that do not consume opaque integer resources.", func() {
		defer removeOpaqueResource(f, node.Name, opaqueResName)

		By("Creating a vanilla pod")
		requests := v1.ResourceList{v1.ResourceCPU: resource.MustParse("0.1")}
		limits := v1.ResourceList{v1.ResourceCPU: resource.MustParse("0.2")}
		pod := f.NewTestPod("without-oir", requests, limits)

		By("Observing an event that indicates the pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		// Here we don't check for the bound node name since it can land on
		// any one (this pod doesn't require any of the opaque resource.)
		predicate := scheduleSuccessEvent(pod.Name, "")
		success, err := common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should schedule pods that do consume opaque integer resources.", func() {
		defer removeOpaqueResource(f, node.Name, opaqueResName)

		By("Creating a pod that requires less of the opaque resource than is allocatable on a node.")
		requests := v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("0.1"),
			opaqueResName:  resource.MustParse("1"),
		}
		limits := v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("0.2"),
			opaqueResName:  resource.MustParse("2"),
		}
		pod := f.NewTestPod("min-oir", requests, limits)

		By("Observing an event that indicates the pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			return err
		}
		predicate := scheduleSuccessEvent(pod.Name, node.Name)
		success, err := common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should not schedule pods that exceed the available amount of opaque integer resource.", func() {
		defer removeOpaqueResource(f, node.Name, opaqueResName)

		By("Creating a pod that requires more of the opaque resource than is allocatable on any node")
		requests := v1.ResourceList{opaqueResName: resource.MustParse("6")}
		limits := v1.ResourceList{}

		By("Observing an event that indicates the pod was not scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(f.NewTestPod("over-max-oir", requests, limits))
			return err
		}
		predicate := scheduleFailureEvent("over-max-oir")
		success, err := common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should account opaque integer resources in pods with multiple containers.", func() {
		defer removeOpaqueResource(f, node.Name, opaqueResName)

		By("Creating a pod with two containers that together require less of the opaque resource than is allocatable on a node")
		requests := v1.ResourceList{opaqueResName: resource.MustParse("1")}
		limits := v1.ResourceList{}
		image := framework.GetPauseImageName(f.ClientSet)
		// This pod consumes 2 "foo" resources.
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "mult-container-oir",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: image,
						Resources: v1.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
					{
						Name:  "pause-sidecar",
						Image: image,
						Resources: v1.ResourceRequirements{
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
		predicate := scheduleSuccessEvent(pod.Name, node.Name)
		success, err := common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))

		By("Creating a pod with two containers that together require more of the opaque resource than is allocatable on any node")
		requests = v1.ResourceList{opaqueResName: resource.MustParse("3")}
		limits = v1.ResourceList{}
		// This pod consumes 6 "foo" resources.
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "mult-container-over-max-oir",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: image,
						Resources: v1.ResourceRequirements{
							Requests: requests,
							Limits:   limits,
						},
					},
					{
						Name:  "pause-sidecar",
						Image: image,
						Resources: v1.ResourceRequirements{
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
		predicate = scheduleFailureEvent(pod.Name)
		success, err = common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})

	It("should schedule pods that initially do not fit after enough opaque integer resources are freed.", func() {
		defer removeOpaqueResource(f, node.Name, opaqueResName)

		By("Creating a pod that requires less of the opaque resource than is allocatable on a node.")
		requests := v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("0.1"),
			opaqueResName:  resource.MustParse("3"),
		}
		limits := v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("0.2"),
			opaqueResName:  resource.MustParse("3"),
		}
		pod1 := f.NewTestPod("oir-1", requests, limits)
		pod2 := f.NewTestPod("oir-2", requests, limits)

		By("Observing an event that indicates one pod was scheduled")
		action := func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod1)
			return err
		}
		predicate := scheduleSuccessEvent(pod1.Name, node.Name)
		success, err := common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))

		By("Observing an event that indicates a subsequent pod was not scheduled")
		action = func() error {
			_, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod2)
			return err
		}
		predicate = scheduleFailureEvent(pod2.Name)
		success, err = common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))

		By("Observing an event that indicates the second pod was scheduled after deleting the first pod")
		action = func() error {
			err := f.ClientSet.Core().Pods(f.Namespace.Name).Delete(pod1.Name, nil)
			return err
		}
		predicate = scheduleSuccessEvent(pod2.Name, node.Name)
		success, err = common.ObserveEventAfterAction(f, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))
	})
})

// Adds the opaque resource to a node.
func addOpaqueResource(f *framework.Framework, nodeName string, opaqueResName v1.ResourceName) {
	action := func() error {
		By(fmt.Sprintf("Adding OIR to node [%s]", nodeName))
		patch := []byte(fmt.Sprintf(`[{"op": "add", "path": "/status/capacity/%s", "value": "5"}]`, escapeForJSONPatch(opaqueResName)))
		return f.ClientSet.Core().RESTClient().Patch(types.JSONPatchType).Resource("nodes").Name(nodeName).SubResource("status").Body(patch).Do().Error()
	}
	predicate := func(n *v1.Node) bool {
		capacity, foundCap := n.Status.Capacity[opaqueResName]
		allocatable, foundAlloc := n.Status.Allocatable[opaqueResName]
		By(fmt.Sprintf("Node [%s] has OIR capacity: [%t] (%s), has OIR allocatable: [%t] (%s)", n.Name, foundCap, capacity.String(), foundAlloc, allocatable.String()))
		return foundCap && capacity.MilliValue() == int64(5000) &&
			foundAlloc && allocatable.MilliValue() == int64(5000)
	}
	success, err := common.ObserveNodeUpdateAfterAction(f, nodeName, predicate, action)
	Expect(err).NotTo(HaveOccurred())
	Expect(success).To(Equal(true))
}

// Removes the opaque resource from a node.
func removeOpaqueResource(f *framework.Framework, nodeName string, opaqueResName v1.ResourceName) {
	action := func() error {
		By(fmt.Sprintf("Removing OIR from node [%s]", nodeName))
		patch := []byte(fmt.Sprintf(`[{"op": "remove", "path": "/status/capacity/%s"}]`, escapeForJSONPatch(opaqueResName)))
		f.ClientSet.Core().RESTClient().Patch(types.JSONPatchType).Resource("nodes").Name(nodeName).SubResource("status").Body(patch).Do()
		return nil // Ignore error -- the opaque resource may not exist.
	}
	predicate := func(n *v1.Node) bool {
		capacity, foundCap := n.Status.Capacity[opaqueResName]
		allocatable, foundAlloc := n.Status.Allocatable[opaqueResName]
		By(fmt.Sprintf("Node [%s] has OIR capacity: [%t] (%s), has OIR allocatable: [%t] (%s)", n.Name, foundCap, capacity.String(), foundAlloc, allocatable.String()))
		return (!foundCap || capacity.IsZero()) && (!foundAlloc || allocatable.IsZero())
	}
	success, err := common.ObserveNodeUpdateAfterAction(f, nodeName, predicate, action)
	Expect(err).NotTo(HaveOccurred())
	Expect(success).To(Equal(true))
}

func escapeForJSONPatch(resName v1.ResourceName) string {
	// Escape forward slashes in the resource name per the JSON Pointer spec.
	// See https://tools.ietf.org/html/rfc6901#section-3
	return strings.Replace(string(resName), "/", "~1", -1)
}
