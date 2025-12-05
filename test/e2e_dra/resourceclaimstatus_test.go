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
	"bytes"
	"encoding/json"

	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	resourceapiv1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	resourceapiac "k8s.io/client-go/applyconfigurations/resource/v1"
	resourceapiacv1beta2 "k8s.io/client-go/applyconfigurations/resource/v1beta2"
	draapiv1beta2 "k8s.io/dynamic-resource-allocation/api/v1beta2"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// resourceClaimDeviceStatus corresponds to testResourceClaimDeviceStatus in test/integration/dra
// and was copied from there, therefore the unit-test style with tCtx and require.
// This is the preferred style for new tests.
func resourceClaimDeviceStatus(tCtx ktesting.TContext, f *framework.Framework, b *drautils.Builder) step2Func {
	namespace := f.Namespace.Name
	claimName := "claim-with-device-status"
	claim := &resourceapiv1beta2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      claimName,
		},
		Spec: resourceapiv1beta2.ResourceClaimSpec{
			Devices: resourceapiv1beta2.DeviceClaim{
				Requests: []resourceapiv1beta2.DeviceRequest{
					{
						Name: "foo",
						Exactly: &resourceapiv1beta2.ExactDeviceRequest{
							DeviceClassName: "foo",
						},
					},
				},
			},
		},
	}

	claim, err := tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create ResourceClaim")

	// Add an allocation result.
	// A finalizer is required for that.
	finalizer := "test.example.com/my-test-finalizer"
	claim.Finalizers = append(claim.Finalizers, finalizer)
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).Update(tCtx, claim, metav1.UpdateOptions{})
	claim.Status.Allocation = &resourceapiv1beta2.AllocationResult{
		Devices: resourceapiv1beta2.DeviceAllocationResult{
			Results: []resourceapiv1beta2.DeviceRequestAllocationResult{
				{
					Request: "foo",
					Driver:  "one",
					Pool:    "global",
					Device:  "my-device",
				},
				{
					Request: "foo",
					Driver:  "two",
					Pool:    "global",
					Device:  "another-device",
				},
				{
					Request: "foo",
					Driver:  "three",
					Pool:    "global",
					Device:  "my-device",
				},
			},
		},
	}
	tCtx.ExpectNoError(err, "add finalizer")
	removeClaim := func(tCtx ktesting.TContext) {
		client := tCtx.Client().ResourceV1beta2()
		claim, err := client.ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		tCtx.ExpectNoError(err, "get claim to remove finalizer")
		if claim.Status.Allocation != nil {
			claim.Status.Allocation = nil
			claim, err = client.ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "remove allocation")
		}
		claim.Finalizers = nil
		claim, err = client.ResourceClaims(namespace).Update(tCtx, claim, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, "remove finalizer")
		err = client.ResourceClaims(namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete claim")
	}
	tCtx.CleanupCtx(removeClaim)
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add allocation result")

	// Now adding the device status should work.
	deviceStatus := []resourceapiv1beta2.AllocatedDeviceStatus{{
		Driver: "one",
		Pool:   "global",
		Device: "my-device",
		Data: &apiruntime.RawExtension{
			Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
		},
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-1",
			IPs: []string{
				"10.9.8.0/24",
				"2001:db8::/64",
			},
			HardwareAddress: "ea:9f:cb:40:b1:7b",
		},
	}}
	claim.Status.Devices = deviceStatus
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after adding device status")

	// Strip the RawExtension. SSA re-encodes it, which causes negligble differences that nonetheless break assert.Equal.
	claim.Status.Devices[0].Data = nil
	deviceStatus[0].Data = nil
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after stripping RawExtension")

	// Exercise SSA.
	deviceStatusAC := resourceapiacv1beta2.AllocatedDeviceStatus().
		WithDriver("two").
		WithPool("global").
		WithDevice("another-device").
		WithNetworkData(resourceapiacv1beta2.NetworkDeviceData().WithInterfaceName("net-2"))
	deviceStatus = append(deviceStatus, resourceapiv1beta2.AllocatedDeviceStatus{
		Driver: "two",
		Pool:   "global",
		Device: "another-device",
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-2",
		},
	})
	claimAC := resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiacv1beta2.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-1",
	})
	tCtx.ExpectNoError(err, "apply device status two")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status two")

	deviceStatusAC = resourceapiacv1beta2.AllocatedDeviceStatus().
		WithDriver("three").
		WithPool("global").
		WithDevice("my-device").
		WithNetworkData(resourceapiacv1beta2.NetworkDeviceData().WithInterfaceName("net-3"))
	deviceStatus = append(deviceStatus, resourceapiv1beta2.AllocatedDeviceStatus{
		Driver: "three",
		Pool:   "global",
		Device: "my-device",
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-3",
		},
	})
	claimAC = resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiacv1beta2.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-2",
	})
	tCtx.ExpectNoError(err, "apply device status three")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status three")
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetIndent("   ", "   ")
	tCtx.ExpectNoError(encoder.Encode(claim))
	tCtx.Logf("Final ResourceClaim:\n%s", buffer.String())

	return func(tCtx ktesting.TContext) step3Func {
		// Update one entry, remove the other.
		deviceStatusAC := resourceapiac.AllocatedDeviceStatus().
			WithDriver("two").
			WithPool("global").
			WithDevice("another-device").
			WithNetworkData(resourceapiac.NetworkDeviceData().WithInterfaceName("yet-another-net"))
		deviceStatus[1].NetworkData.InterfaceName = "yet-another-net"
		claimAC := resourceapiac.ResourceClaim(claim.Name, claim.Namespace).
			WithStatus(resourceapiac.ResourceClaimStatus().WithDevices(deviceStatusAC))
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
			Force:        true,
			FieldManager: "manager-1",
		})
		tCtx.ExpectNoError(err, "update device status two")

		var deviceStatusV1 []resourceapi.AllocatedDeviceStatus
		for _, status := range deviceStatus {
			var statusV1 resourceapi.AllocatedDeviceStatus
			tCtx.ExpectNoError(draapiv1beta2.Convert_v1beta2_AllocatedDeviceStatus_To_v1_AllocatedDeviceStatus(&status, &statusV1, nil))
			deviceStatusV1 = append(deviceStatusV1, statusV1)
		}
		require.Equal(tCtx, deviceStatusV1, claim.Status.Devices, "after updating device status two")

		return func(tCtx ktesting.TContext) {
			claimAC := resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace)
			deviceStatus = deviceStatus[0:2]
			claim, err := tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
				Force:        true,
				FieldManager: "manager-2",
			})
			tCtx.ExpectNoError(err, "remove device status three")
			require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after removing device status three")

			// The cleanup order is so that we have to run this explicitly now.
			// The tCtx.CleanupCtx is more for the sake of completeness.
			removeClaim(tCtx)
		}
	}
}
