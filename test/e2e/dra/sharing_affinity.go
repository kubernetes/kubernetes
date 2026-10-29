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
	"fmt"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/ktesting"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/kubernetes/pkg/features"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = framework.SIGDescribe("node")(feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DRASharingAffinity), framework.WithFeatureGate(features.DRAConsumableCapacity), "DRA sharing affinity", func() {
	f := framework.NewDefaultFramework("dra-sharing-affinity")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	const (
		subnetAPIGroup   = "networking.example.com"
		nicConfigKind    = "NICConfig"
		vlanConfigKind   = "VLANConfig"
		subnetExtractor  = "parameters.subnetID"
		vlanExtractor    = "parameters.vlanID"
		capacitySlotName = resourceapi.QualifiedName("slots")
	)

	singleKeyExtractors := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "subnet",
			APIGroup:   subnetAPIGroup,
			Kind:       nicConfigKind,
			Expression: subnetExtractor,
		},
	}

	multiKeyExtractors := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "subnet",
			APIGroup:   subnetAPIGroup,
			Kind:       nicConfigKind,
			Expression: subnetExtractor,
		},
		{
			Name:       "vlan",
			APIGroup:   subnetAPIGroup,
			Kind:       vlanConfigKind,
			Expression: vlanExtractor,
		},
	}

	shareableDevices := func(count int) []resourceapi.Device {
		devices := make([]resourceapi.Device, count)
		for i := range count {
			devices[i] = resourceapi.Device{
				Name:                     fmt.Sprintf("nic-%d", i),
				AllowMultipleAllocations: ptr.To(true),
				Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					capacitySlotName: {
						Value: resource.MustParse("4"),
						RequestPolicy: &resourceapi.CapacityRequestPolicy{
							Default: ptr.To(resource.MustParse("1")),
							ValidRange: &resourceapi.CapacityRequestPolicyRange{
								Min: ptr.To(resource.MustParse("1")),
							},
						},
					},
				},
			}
		}
		return devices
	}

	poolWithSharingAffinity := func(generation int64, extractors []resourceapi.SharingAffinityExtractor, devices ...resourceapi.Device) func(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
		return func(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
			nodename := nodes.NodeNames[0]
			var slices []resourceslice.Slice
			if len(extractors) > 0 {
				slices = append(slices, resourceslice.Slice{
					SharingAffinity: extractors,
				})
			}
			slices = append(slices, resourceslice.Slice{
				Devices: devices,
			})
			return map[string]resourceslice.DriverResources{
				nodename: {
					Pools: map[string]resourceslice.Pool{
						nodename: {
							Generation: generation,
							Slices:     slices,
						},
					},
				},
			}
		}
	}

	nicConfigRaw := func(subnetID string) runtime.RawExtension {
		return runtime.RawExtension{
			Raw: []byte(fmt.Sprintf(`{"apiVersion":"%s/v1","kind":%q,"subnetID":%q}`, subnetAPIGroup, nicConfigKind, subnetID)),
		}
	}

	vlanConfigRaw := func(vlanID string) runtime.RawExtension {
		return runtime.RawExtension{
			Raw: []byte(fmt.Sprintf(`{"apiVersion":"%s/v1","kind":%q,"vlanID":%q}`, subnetAPIGroup, vlanConfigKind, vlanID)),
		}
	}

	claimWithConfigs := func(b *drautils.Builder, rawConfigs ...runtime.RawExtension) *resourceapi.ResourceClaim {
		claim := b.ExternalClaim()
		claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]resource.Quantity{
				capacitySlotName: resource.MustParse("1"),
			},
		}
		claim.Spec.Devices.Config = make([]resourceapi.DeviceClaimConfiguration, 0, len(rawConfigs))
		for _, raw := range rawConfigs {
			claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, resourceapi.DeviceClaimConfiguration{
				Requests: []string{claim.Spec.Devices.Requests[0].Name},
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver:     b.Driver.Name,
						Parameters: raw,
					},
				},
			})
		}
		return claim
	}

	deleteClaimAndWaitForNotFound := func(tCtx ktesting.TContext, b *drautils.Builder, claim *resourceapi.ResourceClaim) {
		tCtx.Helper()
		err := b.ClientV1(tCtx).ResourceClaims(claim.Namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			tCtx.ExpectNoError(err, "delete claim %s", claim.Name)
		}
		tCtx.Eventually(func(tCtx ktesting.TContext) (bool, error) {
			_, err := b.ClientV1(tCtx).ResourceClaims(claim.Namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}).WithTimeout(time.Minute).Should(gomega.BeTrue(), "expected claim %s to be deleted", claim.Name)
	}

	allocatedDeviceName := func(tCtx ktesting.TContext, b *drautils.Builder, claimName string) string {
		tCtx.Helper()
		claim, err := b.ClientV1(tCtx).ResourceClaims(f.Namespace.Name).Get(tCtx, claimName, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get claim %s", claimName)
		tCtx.Expect(claim.Status.Allocation).NotTo(gomega.BeNil(), "claim %s should be allocated", claimName)
		tCtx.Expect(claim.Status.Allocation.Devices.Results).To(gomega.HaveLen(1), "claim %s should have 1 device result", claimName)
		tCtx.Expect(claim.Status.Allocation.Devices.Results[0].ShareID).NotTo(gomega.BeNil(), "claim %s result should have a ShareID", claimName)
		return claim.Status.Allocation.Devices.Results[0].Device
	}

	f.Context("single shareable device", func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, poolWithSharingAffinity(1, singleKeyExtractors, shareableDevices(1)...))
		b := drautils.NewBuilder(f, driver)

		f.It("co-locates matching affinity, blocks conflicting affinity while locked, and re-locks after full drain", f.WithKubeletMinVersion("1.34"), func(ctx context.Context) {
			tCtx := f.TContext(ctx)

			// 1. Two pods with matching affinity (subnet-a) schedule and share nic-0.
			claimA1 := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			podA1 := b.PodExternal(claimA1.Name)
			b.Create(tCtx, claimA1, podA1)
			b.TestPod(tCtx, podA1, "user_subnetID", "subnet-a")
			devA1 := allocatedDeviceName(tCtx, b, claimA1.Name)

			claimA2 := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			podA2 := b.PodExternal(claimA2.Name)
			b.Create(tCtx, claimA2, podA2)
			b.TestPod(tCtx, podA2, "user_subnetID", "subnet-a")
			devA2 := allocatedDeviceName(tCtx, b, claimA2.Name)
			tCtx.Expect(devA2).To(gomega.Equal(devA1), "both subnet-a claims must share the same device")

			// 2. A third pod with conflicting affinity (subnet-b) is blocked while nic-0 is locked to subnet-a.
			claimB1 := claimWithConfigs(b, nicConfigRaw("subnet-b"))
			podB1 := b.PodExternal(claimB1.Name)
			b.Create(tCtx, claimB1, podB1)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, podB1.Name, podB1.Namespace))

			// 3. Deleting only the first subnet-a pod/claim keeps nic-0 locked because claimA2 is still active.
			b.DeletePodAndWaitForNotFound(tCtx, podA1)
			deleteClaimAndWaitForNotFound(tCtx, b, claimA1)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				gotPod, err := f.ClientSet.CoreV1().Pods(podB1.Namespace).Get(ctx, podB1.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if gotPod.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s expected Pending while device is still locked, got %s", podB1.Name, gotPod.Status.Phase)
				}
				return nil
			}, 5*time.Second, 200*time.Millisecond).Should(gomega.Succeed())

			// 4. Deleting the remaining subnet-a pod/claim drains nic-0, clearing the lock so podB1 schedules and re-locks nic-0 to subnet-b.
			b.DeletePodAndWaitForNotFound(tCtx, podA2)
			deleteClaimAndWaitForNotFound(tCtx, b, claimA2)

			b.TestPod(tCtx, podB1, "user_subnetID", "subnet-b")
			devB1 := allocatedDeviceName(tCtx, b, claimB1.Name)
			tCtx.Expect(devB1).To(gomega.Equal(devA1), "subnet-b claim should allocate the drained device")
		})
	})

	f.Context("multiple shareable devices", func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, poolWithSharingAffinity(1, singleKeyExtractors, shareableDevices(2)...))
		b := drautils.NewBuilder(f, driver)

		f.It("isolates conflicting affinity values onto separate devices while co-locating matching claims", f.WithKubeletMinVersion("1.34"), func(ctx context.Context) {
			tCtx := f.TContext(ctx)

			claimA1 := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			podA1 := b.PodExternal(claimA1.Name)
			b.Create(tCtx, claimA1, podA1)
			b.TestPod(tCtx, podA1, "user_subnetID", "subnet-a")

			claimB1 := claimWithConfigs(b, nicConfigRaw("subnet-b"))
			podB1 := b.PodExternal(claimB1.Name)
			b.Create(tCtx, claimB1, podB1)
			b.TestPod(tCtx, podB1, "user_subnetID", "subnet-b")

			claimA2 := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			podA2 := b.PodExternal(claimA2.Name)
			b.Create(tCtx, claimA2, podA2)
			b.TestPod(tCtx, podA2, "user_subnetID", "subnet-a")

			devA1 := allocatedDeviceName(tCtx, b, claimA1.Name)
			devB1 := allocatedDeviceName(tCtx, b, claimB1.Name)
			devA2 := allocatedDeviceName(tCtx, b, claimA2.Name)

			tCtx.Expect(devA1).To(gomega.Equal(devA2), "matching subnet-a claims should co-locate on the same device")
			tCtx.Expect(devB1).NotTo(gomega.Equal(devA1), "conflicting subnet-b claim must be isolated onto the second device")
		})
	})

	f.Context("rollout with existing allocations", func() {
		nodes := drautils.NewNodes(f, 1, 1)
		// Start pool at generation 1 without sharingAffinity.
		driver := drautils.NewDriver(f, nodes, poolWithSharingAffinity(1, nil, shareableDevices(1)...))
		b := drautils.NewBuilder(f, driver)

		f.It("preserves running legacy pod when sharingAffinity is added, blocks sharing with unreconstructable allocation, and recovers after drain", f.WithKubeletMinVersion("1.34"), func(ctx context.Context) {
			tCtx := f.TContext(ctx)

			// 1. Schedule and run a legacy pod whose claim parameters do not match NICConfig.
			legacyRaw := runtime.RawExtension{Raw: []byte(`{"a":"b"}`)}
			legacyClaim := claimWithConfigs(b, legacyRaw)
			legacyPod := b.PodExternal(legacyClaim.Name)
			b.Create(tCtx, legacyClaim, legacyPod)
			b.TestPod(tCtx, legacyPod, "user_a", "b")

			// 2. Roll out sharingAffinity on the pool (generation 2).
			updatedResources := poolWithSharingAffinity(2, singleKeyExtractors, shareableDevices(1)...)(nodes)
			driver.PublishResources(tCtx, updatedResources)

			tCtx.Eventually(func(tCtx ktesting.TContext) (bool, error) {
				slices, err := tCtx.Client().ResourceV1().ResourceSlices().List(tCtx, metav1.ListOptions{
					FieldSelector: fmt.Sprintf("spec.driver=%s", driver.Name),
				})
				if err != nil {
					return false, err
				}
				hasAffinitySlice := false
				allGen2 := len(slices.Items) == 2
				for _, s := range slices.Items {
					if s.Spec.Pool.Generation != 2 {
						allGen2 = false
					}
					if len(s.Spec.SharingAffinity) > 0 {
						hasAffinitySlice = true
					}
				}
				return allGen2 && hasAffinitySlice, nil
			}).WithTimeout(time.Minute).Should(gomega.BeTrue(), "expected pool generation 2 with sharingAffinity slice")

			// 3. Verify the existing legacy pod remains Running.
			gotLegacyPod, err := f.ClientSet.CoreV1().Pods(legacyPod.Namespace).Get(ctx, legacyPod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get legacy pod")
			tCtx.Expect(gotLegacyPod.Status.Phase).To(gomega.Equal(v1.PodRunning), "legacy pod should continue running after pool update")

			// 4. A new pod with valid NICConfig cannot share nic-0 while the unreconstructable legacy claim holds it.
			newClaim := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			newPod := b.PodExternal(newClaim.Name)
			b.Create(tCtx, newClaim, newPod)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, newPod.Name, newPod.Namespace))

			// 5. Once the legacy pod and claim are removed, nic-0 becomes allocatable again and newPod runs.
			b.DeletePodAndWaitForNotFound(tCtx, legacyPod)
			deleteClaimAndWaitForNotFound(tCtx, b, legacyClaim)

			b.TestPod(tCtx, newPod, "user_subnetID", "subnet-a")
		})
	})

	f.Context("multi-key extraction and strict gating", func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, poolWithSharingAffinity(1, multiKeyExtractors, shareableDevices(1)...))
		b := drautils.NewBuilder(f, driver)

		f.It("enforces all extractor keys, rejects self-inconsistent claims, and requires all keys to match for sharing", f.WithKubeletMinVersion("1.34"), func(ctx context.Context) {
			tCtx := f.TContext(ctx)

			// 1. Strict gating: claim providing only subnet (missing vlan) cannot allocate from the pool.
			incompleteClaim := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			incompletePod := b.PodExternal(incompleteClaim.Name)
			b.Create(tCtx, incompleteClaim, incompletePod)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, incompletePod.Name, incompletePod.Namespace))
			b.DeletePodAndWaitForNotFound(tCtx, incompletePod)
			deleteClaimAndWaitForNotFound(tCtx, b, incompleteClaim)

			// 2. Self-inconsistent claim: two NICConfig entries yielding different subnet values cannot allocate.
			inconsistentClaim := claimWithConfigs(b, nicConfigRaw("subnet-a"), nicConfigRaw("subnet-b"), vlanConfigRaw("100"))
			inconsistentPod := b.PodExternal(inconsistentClaim.Name)
			b.Create(tCtx, inconsistentClaim, inconsistentPod)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, inconsistentPod.Name, inconsistentPod.Namespace))
			b.DeletePodAndWaitForNotFound(tCtx, inconsistentPod)
			deleteClaimAndWaitForNotFound(tCtx, b, inconsistentClaim)

			// 3. Multi-key matching claims (subnet-a + vlan 100) schedule and share nic-0,
			// including when one key is supplied via DeviceClass.spec.config and the other via ResourceClaim.spec.devices.config.
			claim1 := claimWithConfigs(b, nicConfigRaw("subnet-a"), vlanConfigRaw("100"))
			pod1 := b.PodExternal(claim1.Name)
			b.Create(tCtx, claim1, pod1)
			b.TestPod(tCtx, pod1, "user_subnetID", "subnet-a", "user_vlanID", "100")

			classWithVLAN := &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: f.UniqueName + "-vlan-class",
				},
				Spec: resourceapi.DeviceClassSpec{
					Selectors: []resourceapi.DeviceSelector{
						{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf("device.driver == %q", driver.Name),
							},
						},
					},
					Config: []resourceapi.DeviceClassConfiguration{
						{
							DeviceConfiguration: resourceapi.DeviceConfiguration{
								Opaque: &resourceapi.OpaqueDeviceConfiguration{
									Driver:     driver.Name,
									Parameters: vlanConfigRaw("100"),
								},
							},
						},
					},
				},
			}
			b.Create(tCtx, classWithVLAN)
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				_ = b.ClientV1(tCtx).DeviceClasses().Delete(tCtx, classWithVLAN.Name, metav1.DeleteOptions{})
			})

			claim2 := claimWithConfigs(b, nicConfigRaw("subnet-a"))
			claim2.Spec.Devices.Requests[0].Exactly.DeviceClassName = classWithVLAN.Name
			pod2 := b.PodExternal(claim2.Name)
			b.Create(tCtx, claim2, pod2)
			b.TestPod(tCtx, pod2, "user_subnetID", "subnet-a", "admin_vlanID", "100")

			// 4. A claim matching subnet-a but differing on vlan (200 vs 100) is blocked.
			mismatchClaim := claimWithConfigs(b, nicConfigRaw("subnet-a"), vlanConfigRaw("200"))
			mismatchPod := b.PodExternal(mismatchClaim.Name)
			b.Create(tCtx, mismatchClaim, mismatchPod)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, mismatchPod.Name, mismatchPod.Namespace))
		})
	})
})
