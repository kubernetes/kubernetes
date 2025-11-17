/*
Copyright 2024 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("PodRejectionStatus", func() {
	f := framework.NewDefaultFramework("pod-rejection-status")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("Kubelet", func() {
		ginkgo.It("should reject pod when the node didn't have enough resource", func(ctx context.Context) {
			node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
			framework.ExpectNoError(err, "Failed to get a ready schedulable node")

			// Create a pod that requests more CPU than the node has
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-out-of-cpu",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "pod-out-of-cpu",
							Image: imageutils.GetPauseImageName(),
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("1000000000000"), // requests more CPU than any node has
								},
							},
						},
					},
				},
			}

			pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			// Wait for the scheduler to update the pod status
			err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
			framework.ExpectNoError(err)

			// Fetch the pod to get the latest status which should be last one observed by the scheduler
			// before it rejected the pod
			pod, err = f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// force assign the Pod to a node in order to get rejection status later
			binding := &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pod.Name,
					Namespace: pod.Namespace,
					UID:       pod.UID,
				},
				Target: v1.ObjectReference{
					Kind: "Node",
					Name: node.Name,
				},
			}
			err = f.ClientSet.CoreV1().Pods(pod.Namespace).Bind(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			// kubelet has rejected the pod
			err = e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, "OutOfcpu", f.Timeouts.PodStartShort)
			framework.ExpectNoError(err)

			// fetch the reject Pod and compare the status
			gotPod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// This detects if there are any new fields in Status that were dropped by the pod rejection.
			// These new fields either should be kept by kubelet's admission or added explicitly in the list of fields that are having a different value or must be cleared.
			gomega.Expect(gotPod.Status).To(gstruct.MatchAllFields(gstruct.Fields{
				"ObservedGeneration":          gstruct.Ignore(),
				"Phase":                       gstruct.Ignore(),
				"Conditions":                  gstruct.Ignore(),
				"Message":                     gstruct.Ignore(),
				"Reason":                      gstruct.Ignore(),
				"NominatedNodeName":           gstruct.Ignore(),
				"HostIP":                      gstruct.Ignore(),
				"HostIPs":                     gstruct.Ignore(),
				"PodIP":                       gstruct.Ignore(),
				"PodIPs":                      gstruct.Ignore(),
				"StartTime":                   gstruct.Ignore(),
				"InitContainerStatuses":       gstruct.Ignore(),
				"ContainerStatuses":           gstruct.Ignore(),
				"QOSClass":                    gomega.Equal(pod.Status.QOSClass), // QOSClass should be kept
				"EphemeralContainerStatuses":  gstruct.Ignore(),
				"Resize":                      gstruct.Ignore(),
				"ResourceClaimStatuses":       gstruct.Ignore(),
				"ExtendedResourceClaimStatus": gstruct.Ignore(),
				"Resources":                   gstruct.Ignore(),
				"AllocatedResources":          gstruct.Ignore(),
			}))
		})
	})
})
