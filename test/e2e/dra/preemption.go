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

package dra

import (
	"context"

	"github.com/onsi/ginkgo/v2"

	schedulingv1 "k8s.io/api/scheduling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// This suite covers preemption of pods with ResourceClaims (KEP-5690) with a
// real kubelet, ResourceClaim controller and garbage collector, which stop
// preempted pods and deallocate and delete their claims.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DRAPreemption), func() {
	f := framework.NewDefaultFramework("dra-preemption")

	// The driver containers run privileged to manage /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nodes := drautils.NewNodes(f, 1, 1)
	// One device per node.
	driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
	b := drautils.NewBuilder(f, driver)

	createPriorityClass := func(ctx context.Context, name string, value int32) string {
		priorityClass := &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{Name: f.Namespace.Name + "-" + name},
			Value:      value,
		}
		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, priorityClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "create priority class")
		ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.SchedulingV1().PriorityClasses().Delete), priorityClass.Name, metav1.DeleteOptions{})
		return priorityClass.Name
	}

	f.It("preempts a pod with lower priority which uses the device", func(ctx context.Context) {
		tCtx := f.TContext(ctx)
		low := createPriorityClass(ctx, "low", 10)
		high := createPriorityClass(ctx, "high", 100)

		victim, victimTemplate := b.PodInline()
		victim.Spec.PriorityClassName = low
		b.Create(tCtx, victimTemplate, victim)
		b.TestPod(tCtx, victim)

		preemptor, preemptorTemplate := b.PodInline()
		preemptor.Spec.PriorityClassName = high
		b.Create(tCtx, preemptorTemplate, preemptor)
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, victim.Name, victim.Namespace, f.Timeouts.PodDelete), "preempted pod")
		b.TestPod(tCtx, preemptor)
	})
})
