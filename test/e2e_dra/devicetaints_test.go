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
	resourcealpha "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// deviceTaints checks that:
// - A pod which gets scheduled on the previous release because of a toleration is kept running after an upgrade.
// - A DeviceTaintRule created to evict the pod before a downgrade prevents pod scheduling after a downgrade.
func deviceTaints(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	namespace := tCtx.Namespace()
	taintKey := "devicetaints"
	taintValueFromSlice := "from-slice"
	taintValueFromRule := "from-rule"
	taintedDevice := "tainted-device"

	// We need additional devices which are only used by this test.
	// We achieve that with cluster-scoped devices that start out with
	// a taint.
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "devicetaints",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: b.Driver.Name,
			Pool: resourceapi.ResourcePool{
				Name:               "devicetaints",
				ResourceSliceCount: 1,
			},
			AllNodes: ptr.To(true),
			Devices: []resourceapi.Device{{
				Name: taintedDevice,
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"example.com/type": {
						StringValue: ptr.To("devicetaints"),
					},
				},
				Taints: []resourceapi.DeviceTaint{{
					Key:    taintKey,
					Value:  taintValueFromSlice,
					Effect: resourceapi.DeviceTaintEffectNoSchedule,
				}},
			}},
		},
	}
	_, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err)

	tCtx.Log("The pod wants exactly the tainted device -> not schedulable.")
	claim := b.ExternalClaim()
	pod := b.PodExternal()
	pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
	claim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: `device.attributes["example.com"].?type.orValue("") == "devicetaints"`,
		},
	}}
	b.Create(tCtx, claim, pod)
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod.Name, namespace))

	tCtx.Log("Adding a toleration makes the pod schedulable.")
	claim.Spec.Devices.Requests[0].Exactly.Tolerations = []resourceapi.DeviceToleration{{
		Key:    taintKey,
		Value:  taintValueFromSlice,
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	}}
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{}))
	_, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err)
	b.TestPod(tCtx, pod)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		tCtx.Log("Pod running consistently after upgrade.")
		tCtx.Consistently(func(tCtx ktesting.TContext) error {
			return e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod)
		}).WithTimeout(30 * time.Second).WithPolling(5 * time.Second).Should(gomega.Succeed())

		tCtx.Logf("Evict pod through DeviceTaintRule.")
		rule := &resourcealpha.DeviceTaintRule{
			ObjectMeta: metav1.ObjectMeta{
				Name: "device-taint-rule",
			},
			Spec: resourcealpha.DeviceTaintRuleSpec{
				DeviceSelector: &resourcealpha.DeviceTaintSelector{
					Driver: &b.Driver.Name,
					Pool:   &slice.Spec.Pool.Name,
					Device: &taintedDevice,
				},
				Taint: resourcealpha.DeviceTaint{
					Key:    taintKey,
					Value:  taintValueFromRule,
					Effect: resourcealpha.DeviceTaintEffectNoExecute,
				},
			},
		}
		_, err := tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Create(tCtx, rule, metav1.CreateOptions{})
		tCtx.ExpectNoError(err)
		tCtx.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(tCtx, tCtx.Client(), pod.Name, namespace, 5*time.Minute))

		return func(tCtx ktesting.TContext) {
			tCtx.Log("DeviceTaintRule still in effect.")
			b.Create(tCtx, pod)
			tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod.Name, namespace))

			// Clean up manually.
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
		}
	}
}
