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

package kubeletplugin

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/api/metadata/v1alpha1"
	"k8s.io/dynamic-resource-allocation/devicemetadata"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/utils/ptr"
)

const (
	testDriverName            = "example.com"
	testClaimName             = "my-claim"
	testClaimNS               = "default"
	testClaimUID              = types.UID("claim-uid-1234")
	testRequest               = "gpu-request"
	testRequestWithSubrequest = "gpu-request/high-memory"
)

var metadataTypeMeta = metav1.TypeMeta{
	APIVersion: v1alpha1.SchemeGroupVersion.String(),
	Kind:       "DeviceMetadata",
}

func testClaim() *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClaimName,
			Namespace: testClaimNS,
			UID:       testClaimUID,
		},
	}
}

func TestNewMetadataWriterVersionValidation(t *testing.T) {
	bogus := schema.GroupVersion{Group: "bogus", Version: "v99"}

	testcases := map[string]struct {
		versions  []schema.GroupVersion
		expectErr bool
	}{
		"all supported": {
			versions: []schema.GroupVersion{v1alpha1.SchemeGroupVersion},
		},
		"mix of supported and unsupported": {
			versions: []schema.GroupVersion{bogus, v1alpha1.SchemeGroupVersion},
		},
		"all unsupported": {
			versions:  []schema.GroupVersion{bogus},
			expectErr: true,
		},
		"empty list": {
			versions:  []schema.GroupVersion{},
			expectErr: true,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			_, err := newMetadataWriter(testDriverName, t.TempDir(), t.TempDir(), tc.versions)
			if tc.expectErr && err == nil {
				t.Fatal("expected error but got nil")
			}
			if !tc.expectErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func newTestWriter(t *testing.T) (*metadataWriter, string, string) {
	t.Helper()
	pluginDir := t.TempDir()
	cdiDir := t.TempDir()
	w, err := newMetadataWriter(testDriverName, pluginDir, cdiDir, []schema.GroupVersion{v1alpha1.SchemeGroupVersion})
	if err != nil {
		t.Fatalf("newMetadataWriter: %v", err)
	}
	return w, pluginDir, cdiDir
}

func readMetadataFile(t *testing.T, path string) *v1alpha1.DeviceMetadata {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read metadata file: %v", err)
	}
	var dm v1alpha1.DeviceMetadata
	if err := devicemetadata.DecodeMetadataFromStream(json.NewDecoder(bytes.NewReader(data)), &dm); err != nil {
		t.Fatalf("decode metadata file: %v", err)
	}
	return &dm
}

func stringAttrs(attrs map[string]string) *DeviceMetadata {
	da := make(map[string]resourceapi.DeviceAttribute, len(attrs))
	for k, v := range attrs {
		da[k] = resourceapi.DeviceAttribute{StringValue: &v}
	}
	return &DeviceMetadata{Attributes: da}
}

func expectedV1Alpha1Attrs(attrs map[string]string) map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	da := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, len(attrs))
	for k, v := range attrs {
		da[resourceapi.QualifiedName(k)] = resourceapi.DeviceAttribute{StringValue: &v}
	}
	return da
}

func attrsToStrings(attrs map[string]resourceapi.DeviceAttribute) map[string]string {
	out := make(map[string]string, len(attrs))
	for k, v := range attrs {
		if v.StringValue != nil {
			out[k] = *v.StringValue
		}
	}
	return out
}

func expectedRequestMetadata(reqName string, devs []Device) *v1alpha1.DeviceMetadata {
	var expectedDevs []v1alpha1.Device
	for _, d := range devs {
		ed := v1alpha1.Device{
			Driver: testDriverName,
			Pool:   d.PoolName,
			Name:   d.DeviceName,
		}
		if d.Metadata != nil {
			if d.Metadata.Attributes != nil {
				ed.Attributes = expectedV1Alpha1Attrs(
					attrsToStrings(d.Metadata.Attributes))
			}
			ed.NetworkData = d.Metadata.NetworkData
		}
		expectedDevs = append(expectedDevs, ed)
	}
	return &v1alpha1.DeviceMetadata{
		TypeMeta: metadataTypeMeta,
		ObjectMeta: metav1.ObjectMeta{
			Name:       testClaimName,
			Namespace:  testClaimNS,
			UID:        testClaimUID,
			Generation: 1,
		},
		Requests: []v1alpha1.DeviceMetadataRequest{{
			Name:    reqName,
			Devices: expectedDevs,
		}},
	}
}

// expectedCDISpec returns the expected CDI spec for a single request.
// containerPath is the full expected path inside the container.
func expectedCDISpec(reqName, metadataHostPath, containerPath string) cdiSpec {
	return cdiSpec{
		Version: cdiVersionStr,
		Kind:    testDriverName + "/metadata",
		Devices: []cdiDevice{{
			Name: string(testClaimUID) + "_" + reqName,
			ContainerEdits: cdiContainerEdits{
				Mounts: []cdiMount{{
					HostPath:      metadataHostPath,
					ContainerPath: containerPath,
					Options:       []string{"ro", "bind"},
				}},
			},
		}},
	}
}

type expectedRequestPaths struct {
	hostMetadataSuffix string
	cdiSpecFileName    string
	containerPath      string
}

func claimPaths(requestRef string) expectedRequestPaths {
	baseReq := resourceclaim.BaseRequestRef(requestRef)
	return expectedRequestPaths{
		hostMetadataSuffix: metadataSubDir + "/" + testClaimNS + "_" + testClaimName + "/" + baseReq + "/metadata.json",
		cdiSpecFileName:    testDriverName + "_metadata_" + string(testClaimUID) + "_" + baseReq + ".json",
		containerPath:      metadata.ContainerDir + "/" + metadata.ResourceClaimsSubDir + "/" + testClaimName + "/" + baseReq + "/" + metadata.MetadataFileName(testDriverName),
	}
}

func templateClaimPaths(requestRef, podClaimName string) expectedRequestPaths {
	baseReq := resourceclaim.BaseRequestRef(requestRef)
	return expectedRequestPaths{
		hostMetadataSuffix: metadataSubDir + "/" + testClaimNS + "_" + testClaimName + "/" + baseReq + "/metadata.json",
		cdiSpecFileName:    testDriverName + "_metadata_" + string(testClaimUID) + "_" + baseReq + ".json",
		containerPath:      metadata.ContainerDir + "/" + metadata.ResourceClaimTemplatesSubDir + "/" + podClaimName + "/" + baseReq + "/" + metadata.MetadataFileName(testDriverName),
	}
}

func TestProcessPreparedClaim(t *testing.T) {
	testcases := map[string]struct {
		devices            []Device
		claimAnnotations   map[string]string
		expectPodClaimName *string
		wantPaths          map[string]expectedRequestPaths
	}{
		"with-metadata": {
			devices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
				Metadata:   stringAttrs(map[string]string{"pciBusID": "0000:03:00.0"}),
			}},
			wantPaths: map[string]expectedRequestPaths{
				testRequest: claimPaths(testRequest),
			},
		},
		"without-metadata-deferred": {
			devices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
			}},
			wantPaths: map[string]expectedRequestPaths{
				testRequest: claimPaths(testRequest),
			},
		},
		"network-device-data": {
			devices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "vf-0",
				Metadata: &DeviceMetadata{
					NetworkData: &resourceapi.NetworkDeviceData{
						InterfaceName: "eth0",
						IPs:           []string{"10.0.0.5/24"},
					},
				},
			}},
			wantPaths: map[string]expectedRequestPaths{
				testRequest: claimPaths(testRequest),
			},
		},
		"multiple-requests": {
			devices: []Device{
				{Requests: []string{"gpu-request"}, PoolName: "node-1", DeviceName: "gpu-0"},
				{Requests: []string{"nic-request"}, PoolName: "node-1", DeviceName: "nic-0"},
			},
			wantPaths: map[string]expectedRequestPaths{
				"gpu-request": claimPaths("gpu-request"),
				"nic-request": claimPaths("nic-request"),
			},
		},
		"subrequest": {
			devices: []Device{{
				Requests:   []string{testRequestWithSubrequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
				Metadata:   stringAttrs(map[string]string{"memoryGB": "80"}),
			}},
			wantPaths: map[string]expectedRequestPaths{
				testRequestWithSubrequest: claimPaths(testRequestWithSubrequest),
			},
		},
		"with-pod-claim-name-annotation": {
			devices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
				Metadata:   stringAttrs(map[string]string{"model": "LATEST-GPU-MODEL"}),
			}},
			claimAnnotations:   map[string]string{resourceapi.PodResourceClaimAnnotation: "my-gpu"},
			expectPodClaimName: ptr.To("my-gpu"), //nolint:modernize
			wantPaths: map[string]expectedRequestPaths{
				testRequest: templateClaimPaths(testRequest, "my-gpu"),
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			w, pluginDir, cdiDir := newTestWriter(t)
			claim := testClaim()

			if tc.claimAnnotations != nil {
				claim.Annotations = tc.claimAnnotations
			}

			cdiIDs, err := w.processPreparedClaim(claim, tc.devices)
			if err != nil {
				t.Fatalf("processPreparedClaim: %v", err)
			}

			devicesByRequest := map[string][]Device{}
			for _, d := range tc.devices {
				for _, req := range d.Requests {
					devicesByRequest[req] = append(devicesByRequest[req], d)
				}
			}

			expectedCDIIDs := make(map[string]string, len(devicesByRequest))
			for requestRef, devs := range devicesByRequest {
				baseReq := resourceclaim.BaseRequestRef(requestRef)
				expectedCDIIDs[baseReq] = testDriverName + "/metadata=" + string(testClaimUID) + "_" + baseReq

				wp := tc.wantPaths[requestRef]

				metadataPath := filepath.Join(pluginDir, wp.hostMetadataSuffix)
				dm := readMetadataFile(t, metadataPath)
				expected := expectedRequestMetadata(requestRef, devs)
				expected.PodClaimName = tc.expectPodClaimName
				if diff := cmp.Diff(expected, dm); diff != "" {
					t.Errorf("metadata for request %q (-want +got):\n%s", requestRef, diff)
				}

				cdiPath := filepath.Join(cdiDir, wp.cdiSpecFileName)
				cdiData, err := os.ReadFile(cdiPath)
				if err != nil {
					t.Fatalf("read CDI spec %s: %v", cdiPath, err)
				}
				var gotCDISpec cdiSpec
				if err := json.Unmarshal(cdiData, &gotCDISpec); err != nil {
					t.Fatalf("unmarshal CDI spec: %v", err)
				}

				if diff := cmp.Diff(expectedCDISpec(baseReq, metadataPath, wp.containerPath), gotCDISpec); diff != "" {
					t.Errorf("CDI spec for request %q (-want +got):\n%s", requestRef, diff)
				}
			}
			if diff := cmp.Diff(expectedCDIIDs, cdiIDs); diff != "" {
				t.Errorf("CDI IDs (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCleanupClaim(t *testing.T) {
	strVal := "0000:03:00.0"

	testcases := map[string]struct {
		prepareDevices []Device
		preCleanup     bool
		claimNamespace string
		claimName      string
		uid            types.UID
	}{
		"removes-files": {
			prepareDevices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
				Metadata: &DeviceMetadata{
					Attributes: map[string]resourceapi.DeviceAttribute{
						"pciBusID": {StringValue: &strVal},
					},
				},
			}},
			claimNamespace: testClaimNS,
			claimName:      testClaimName,
			uid:            testClaimUID,
		},
		"nonexistent-claim-is-noop": {
			claimNamespace: "no-ns",
			claimName:      "no-claim",
			uid:            "unknown-uid",
		},
		"idempotent": {
			prepareDevices: []Device{{
				Requests:   []string{testRequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
				Metadata: &DeviceMetadata{
					Attributes: map[string]resourceapi.DeviceAttribute{
						"pciBusID": {StringValue: &strVal},
					},
				},
			}},
			preCleanup:     true,
			claimNamespace: testClaimNS,
			claimName:      testClaimName,
			uid:            testClaimUID,
		},
		"multiple-requests": {
			prepareDevices: []Device{
				{Requests: []string{"gpu-request"}, PoolName: "node-1", DeviceName: "gpu-0"},
				{Requests: []string{"nic-request"}, PoolName: "node-1", DeviceName: "nic-0"},
			},
			claimNamespace: testClaimNS,
			claimName:      testClaimName,
			uid:            testClaimUID,
		},
		"subrequest": {
			prepareDevices: []Device{{
				Requests:   []string{testRequestWithSubrequest},
				PoolName:   "node-1",
				DeviceName: "gpu-0",
			}},
			claimNamespace: testClaimNS,
			claimName:      testClaimName,
			uid:            testClaimUID,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			w, pluginDir, cdiDir := newTestWriter(t)

			if tc.prepareDevices != nil {
				if _, err := w.processPreparedClaim(testClaim(), tc.prepareDevices); err != nil {
					t.Fatalf("processPreparedClaim: %v", err)
				}
			}
			if tc.preCleanup {
				if err := w.cleanupClaim(tc.claimNamespace, tc.claimName, tc.uid); err != nil {
					t.Fatalf("pre-cleanupClaim: %v", err)
				}
			}

			if err := w.cleanupClaim(tc.claimNamespace, tc.claimName, tc.uid); err != nil {
				t.Fatalf("cleanupClaim: %v", err)
			}

			metadataDir := filepath.Join(pluginDir, metadataSubDir, tc.claimNamespace+"_"+tc.claimName)
			if _, err := os.Stat(metadataDir); !os.IsNotExist(err) {
				t.Errorf("metadata directory %s still exists", metadataDir)
			}
			for _, d := range tc.prepareDevices {
				for _, reqName := range d.Requests {
					baseReq := resourceclaim.BaseRequestRef(reqName)
					cdiPath := filepath.Join(cdiDir,
						testDriverName+"_metadata_"+string(tc.uid)+"_"+baseReq+".json")
					if _, err := os.Stat(cdiPath); !os.IsNotExist(err) {
						t.Errorf("CDI spec file %s still exists", cdiPath)
					}
				}
			}
		})
	}
}

func TestUpdateRequestMetadata(t *testing.T) {
	strVal := "0000:03:00.0"
	devicesWithMetadata := []Device{{
		Requests:   []string{testRequest},
		PoolName:   "node-1",
		DeviceName: "gpu-0",
		Metadata: &DeviceMetadata{
			Attributes: map[string]resourceapi.DeviceAttribute{
				"pciBusID": {StringValue: &strVal},
			},
		},
	}}
	devicesWithoutMetadata := []Device{{
		Requests:   []string{testRequest},
		PoolName:   "node-1",
		DeviceName: "gpu-0",
	}}

	subreqRef := testRequest + "/high-memory"
	subreqDevicesWithout := []Device{{
		Requests:   []string{subreqRef},
		PoolName:   "node-1",
		DeviceName: "gpu-0",
	}}
	subreqDevicesWith := []Device{{
		Requests:   []string{subreqRef},
		PoolName:   "node-1",
		DeviceName: "gpu-0",
		Metadata: &DeviceMetadata{
			Attributes: map[string]resourceapi.DeviceAttribute{
				"pciBusID": {StringValue: &strVal},
			},
		},
	}}

	testcases := map[string]struct {
		prepareDevices     []Device
		preUpdates         int
		claimUID           types.UID
		requestName        string
		updateDevices      []Device
		expectError        string
		expectedGeneration int64
	}{
		"updates-file-and-generation": {
			prepareDevices:     devicesWithoutMetadata,
			claimUID:           testClaimUID,
			requestName:        testRequest,
			updateDevices:      devicesWithMetadata,
			expectedGeneration: 2,
		},
		"unknown-claim": {
			claimUID:      "unknown-uid",
			requestName:   testRequest,
			updateDevices: devicesWithMetadata,
			expectError:   "not found",
		},
		"unknown-request": {
			prepareDevices: devicesWithMetadata,
			claimUID:       testClaimUID,
			requestName:    "nonexistent-request",
			updateDevices:  devicesWithMetadata,
			expectError:    "not found",
		},
		"increments-generation-multiple-times": {
			prepareDevices:     devicesWithoutMetadata,
			preUpdates:         2,
			claimUID:           testClaimUID,
			requestName:        testRequest,
			updateDevices:      devicesWithMetadata,
			expectedGeneration: 4,
		},
		"subrequest-update": {
			prepareDevices:     subreqDevicesWithout,
			claimUID:           testClaimUID,
			requestName:        subreqRef,
			updateDevices:      subreqDevicesWith,
			expectedGeneration: 2,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			w, pluginDir, _ := newTestWriter(t)

			if tc.prepareDevices != nil {
				if _, err := w.processPreparedClaim(testClaim(), tc.prepareDevices); err != nil {
					t.Fatalf("processPreparedClaim: %v", err)
				}
			}
			for i := range tc.preUpdates {
				if err := w.updateRequestMetadata(
					claimRef{namespace: testClaimNS, name: testClaimName, uid: testClaimUID},
					tc.requestName, tc.updateDevices); err != nil {
					t.Fatalf("pre-updateRequestMetadata (%d): %v", i+1, err)
				}
			}

			err := w.updateRequestMetadata(
				claimRef{namespace: testClaimNS, name: testClaimName, uid: tc.claimUID},
				tc.requestName, tc.updateDevices)

			if tc.expectError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectError)
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got: %v", tc.expectError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("updateRequestMetadata: %v", err)
			}

			expected := expectedRequestMetadata(tc.requestName, tc.updateDevices)
			expected.ObjectMeta.Generation = tc.expectedGeneration

			baseReq := resourceclaim.BaseRequestRef(tc.requestName)
			metadataPath := filepath.Join(pluginDir, metadataSubDir,
				testClaimNS+"_"+testClaimName, baseReq, "metadata.json")
			dm := readMetadataFile(t, metadataPath)
			if diff := cmp.Diff(expected, dm); diff != "" {
				t.Errorf("metadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
