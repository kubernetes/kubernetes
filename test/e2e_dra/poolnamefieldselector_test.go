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
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

const poolNameFieldSelectorFallbackPool = "pool-name-field-selector-fallback"

func poolNameFieldSelectorFallbackResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	return map[string]resourceslice.DriverResources{
		nodes.NodeNames[0]: {
			Pools: map[string]resourceslice.Pool{
				poolNameFieldSelectorFallbackPool: {
					AllNodes: true,
					Slices: []resourceslice.Slice{{
						Devices: []resourceapi.Device{{Name: "device-00"}},
					}},
				},
			},
		},
	}
}

// poolNameFieldSelectorFallback verifies that the current ResourceSlice
// controller keeps reconciling after the previous release rejects the
// spec.pool.name field selector. The controller keeps using local filtering
// after upgrade and downgrade.
func poolNameFieldSelectorFallback(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	recreatePoolNameFieldSelectorFallbackSlice(tCtx, b)
	return func(tCtx ktesting.TContext) downgradedTestFunc {
		recreatePoolNameFieldSelectorFallbackSlice(tCtx, b)
		return func(tCtx ktesting.TContext) {
			recreatePoolNameFieldSelectorFallbackSlice(tCtx, b)
		}
	}
}

func recreatePoolNameFieldSelectorFallbackSlice(tCtx ktesting.TContext, b *drautils.Builder) {
	getSlices := b.Driver.NewGetSlices()
	resourceSliceMatcher := func() gomega.OmegaMatcher {
		return gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Spec": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Driver":   gomega.Equal(b.Driver.Name),
				"NodeName": gomega.BeNil(),
				"AllNodes": gomega.Equal(new(true)),
				"Pool": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Name":               gomega.Equal(poolNameFieldSelectorFallbackPool),
					"ResourceSliceCount": gomega.Equal(int64(1)),
				}),
				"Devices": gomega.ConsistOf(resourceapi.Device{Name: "device-00"}),
			}),
		})
	}
	tCtx.Eventually(getSlices).Should(gomega.HaveField("Items", gomega.ConsistOf(resourceSliceMatcher())))
	oldSlice := getSlices(tCtx).Items[0]
	tCtx.ExpectNoError(b.Driver.ClientV1(tCtx).ResourceSlices().Delete(tCtx, oldSlice.Name, metav1.DeleteOptions{}), "delete ResourceSlice %q", oldSlice.Name)
	tCtx.Eventually(getSlices).Should(gomega.HaveField("Items", gomega.ConsistOf(
		gomega.And(
			resourceSliceMatcher(),
			gomega.HaveField("UID", gomega.Not(gomega.Equal(oldSlice.UID))),
		),
	)))
}
