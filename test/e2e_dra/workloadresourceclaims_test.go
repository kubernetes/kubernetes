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

package e2edra

import (
	"time"

	"github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

func workloadResourceClaimsDriverResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	return drautils.DriverResourcesNow(nodes, 1)
}

// workloadResourceClaimsGateCycle implements the DRAWorkloadResourceClaims feature gate ON→OFF→ON cycle test:
//
//   - Phase 0 (gate ON): create a PodGroup with 2 member Pods sharing a
//     ResourceClaim reserved for the PodGroup.
//   - Phase 1 (gate OFF): The ResourceClaim remains reserved for
//     the PodGroup and no new ResourceClaims are created. Pods using claims
//     reserved for a PodGroup can still be deleted.
//   - Phase 2 (gate ON again): A new Pod added to the group can share the
//     ResourceClaim initially allocated in phase 0.
func workloadResourceClaimsGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	namespace := tCtx.Namespace()
	workload, template := b.WorkloadInline()
	podGroup := b.PodGroup(workload, workload.Spec.PodGroupTemplates[0])
	pod := b.GroupedPodWithClaims(podGroup)
	podToDelete := b.GroupedPodWithClaims(podGroup)
	b.Create(tCtx, workload, podGroup, template, pod, podToDelete)
	b.TestPod(tCtx, pod)
	b.TestPod(tCtx, podToDelete)

	podGroup, err := tCtx.Client().SchedulingV1alpha3().PodGroups(namespace).Get(tCtx, podGroup.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get PodGroup")
	tCtx.Expect(podGroup.Status.ResourceClaimStatuses).To(gomega.HaveExactElements(
		gomega.SatisfyAll(
			gomega.HaveField("Name", podGroup.Spec.ResourceClaims[0].Name),
			gomega.HaveField("ResourceClaimName", gomega.HaveValue(gomega.Not(gomega.BeEmpty()))),
		),
	), "PodGroup status is missing generated ResourceClaim name")

	claimName := *podGroup.Status.ResourceClaimStatuses[0].ResourceClaimName
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get generated ResourceClaim")
	reservedFor := claim.Status.ReservedFor
	tCtx.Expect(reservedFor).To(gomega.HaveExactElements(
		resourceapi.ResourceClaimConsumerReference{
			APIGroup: schedulingapi.GroupName,
			Resource: "podgroups",
			Name:     podGroup.Name,
			UID:      podGroup.UID,
		},
	))

	checkClaimsStable := func(tCtx ktesting.TContext) {
		tCtx.Helper()
		tCtx.Consistently(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaimList, error) {
			return tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		}).
			WithPolling(100 * time.Millisecond).
			WithTimeout(15 * time.Second).
			Should(gomega.HaveField("Items", gomega.HaveExactElements(
				gomega.SatisfyAll(
					gomega.HaveField("ObjectMeta.UID", claim.UID),
					gomega.HaveField("Status.ReservedFor", reservedFor),
				),
			)))
	}

	return func(tCtx ktesting.TContext) gateOnAgainFunc {
		// A new Pod in the PodGroup making the same claim as the other Pods
		// will not be able to schedule because the ResourceClaim controller
		// will not associate the PodGroup claim with the Pod when the feature
		// is disabled. Kubernetes 1.36 would have created a new ResourceClaim
		// for the Pod, which we want to verify does *not* happen.
		stuckPod := b.GroupedPodWithClaims(podGroup)
		b.Create(tCtx, stuckPod)
		checkClaimsStable(tCtx)

		// Pods created before or after the feature was disabled should still
		// be able to be deleted.
		b.DeletePodAndWaitForNotFound(tCtx, stuckPod)
		b.DeletePodAndWaitForNotFound(tCtx, podToDelete)

		return func(tCtx ktesting.TContext) {
			newPod := b.GroupedPodWithClaims(podGroup)
			b.Create(tCtx, newPod)
			b.TestPod(tCtx, newPod)
			checkClaimsStable(tCtx)
		}
	}
}
