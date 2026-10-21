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

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/features"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const compatGroupsCounterSet = "gpu"

// compatGroupsCounters is a single counter set with room for two devices (each
// device below consumes half of it), so two devices can be co-allocated to
// separate claims on the same node only when their compatibility groups
// intersect - counters alone never force the second pod to be unschedulable.
func compatGroupsCounters() []resourceapi.CounterSet {
	return []resourceapi.CounterSet{{
		Name:     compatGroupsCounterSet,
		Counters: map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("2")}},
	}}
}

func compatGroupsDevice(name, group string) resourceapi.Device {
	return resourceapi.Device{
		Name: name,
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{{
			CounterSet:          compatGroupsCounterSet,
			Counters:            map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("1")}},
			CompatibilityGroups: []string{group},
		}},
	}
}

// This suite exercises the DRADeviceCompatibilityGroups feature end to end with a
// fake DRA driver that advertises devices declaring compatibility groups on a
// shared counter set. It only runs on clusters where the (alpha) feature gate is
// enabled.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), framework.WithFeatureGate(features.DRADeviceCompatibilityGroups), func() {
	f := framework.NewDefaultFramework("dra-compat-groups")

	// The driver containers run privileged to manage /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nodes := drautils.NewNodes(f, 1, 1)

	f.Context("device compatibility groups", func() {
		f.Context("incompatible devices", func() {
			// Two devices declaring mutually exclusive groups (mig, vgpu) share
			// one counter set.
			driver := drautils.NewDriver(f, nodes, drautils.ToDriverResources(
				compatGroupsCounters(),
				compatGroupsDevice("gpu-mig", "mig"),
				compatGroupsDevice("gpu-vgpu", "vgpu"),
			))
			b := drautils.NewBuilder(f, driver)

			f.It("leave a pod requesting an incompatible device unschedulable", func(ctx context.Context) {
				tCtx := f.TContext(ctx)

				// The first pod takes one of the two devices.
				claim1 := b.ExternalClaim()
				b.Create(tCtx, claim1)
				pod1 := b.PodExternal(claim1.Name)
				b.Create(tCtx, pod1)
				b.TestPod(tCtx, pod1)

				// The only remaining device declares an incompatible group on the
				// same counter set, so the second pod cannot be scheduled even
				// though the counter set still has room for it.
				claim2 := b.ExternalClaim()
				b.Create(tCtx, claim2)
				pod2 := b.PodExternal(claim2.Name)
				b.Create(tCtx, pod2)
				framework.ExpectNoError(
					e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod2.Name, pod2.Namespace),
					"second pod must be unschedulable: only an incompatible device remains on the counter set")
			})
		})

		f.Context("compatible devices", func() {
			// Two devices declaring the same group (mig) share one counter set.
			driver := drautils.NewDriver(f, nodes, drautils.ToDriverResources(
				compatGroupsCounters(),
				compatGroupsDevice("gpu-0", "mig"),
				compatGroupsDevice("gpu-1", "mig"),
			))
			b := drautils.NewBuilder(f, driver)

			f.It("schedule two pods that share a compatibility group", func(ctx context.Context) {
				tCtx := f.TContext(ctx)

				claim1 := b.ExternalClaim()
				b.Create(tCtx, claim1)
				pod1 := b.PodExternal(claim1.Name)
				b.Create(tCtx, pod1)
				b.TestPod(tCtx, pod1)

				// The second device shares the first's group, so it can be
				// co-allocated on the same counter set and its pod also runs.
				claim2 := b.ExternalClaim()
				b.Create(tCtx, claim2)
				pod2 := b.PodExternal(claim2.Name)
				b.Create(tCtx, pod2)
				b.TestPod(tCtx, pod2)
			})
		})
	})
})
