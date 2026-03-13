/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/klog/v2"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("PodOSRejection", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("pod-os-rejection")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("Kubelet", func() {
		ginkgo.It("[LinuxOnly] should reject pod when the node OS doesn't match pod's OS", func(ctx context.Context) {
			linuxNode, err := findLinuxNode(ctx, f)
			framework.ExpectNoError(err)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "wrong-pod-os",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					OS: &v1.PodOS{
						Name: "windows", // explicitly set the pod OS to a wrong but valid value
					},
					Containers: []v1.Container{
						{
							Name:  "wrong-pod-os",
							Image: imageutils.GetPauseImageName(),
						},
					},
					NodeName: linuxNode.Name, // Set the node to an node which doesn't support
				},
			}
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			// Check the pod is still not running
			err = e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, "PodOSNotSupported", f.Timeouts.PodStartShort)
			framework.ExpectNoError(err)
		})
	})
})

// findLinuxNode finds a Linux node that is Ready and Schedulable
func findLinuxNode(ctx context.Context, f *framework.Framework) (v1.Node, error) {
	logger := klog.FromContext(ctx)
	selector := labels.Set{"kubernetes.io/os": "linux"}.AsSelector()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{LabelSelector: selector.String()})

	if err != nil {
		return v1.Node{}, err
	}

	var targetNode v1.Node
	foundNode := false
	for _, n := range nodeList.Items {
		if e2enode.IsNodeReady(logger, &n) && e2enode.IsNodeSchedulable(logger, &n) {
			targetNode = n
			foundNode = true
			break
		}
	}

	if !foundNode {
		e2eskipper.Skipf("Could not find and ready and schedulable Linux nodes")
	}

	return targetNode, nil
}
