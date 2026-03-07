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

package kubeletplugin_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
)

// ExampleDecodeMetadataFromStream demonstrates reading a metadata file that
// may contain multiple API versions as a JSON stream. Decoding into the
// internal type (metadata.DeviceMetadata) is recommended because it is
// stable across API version changes.
func ExampleDecodeMetadataFromStream() {
	fileData := []byte(`{"apiVersion":"metadata.resource.k8s.io/v1alpha1","kind":"DeviceMetadata","metadata":{"name":"my-claim","namespace":"default","uid":"uid-1234","generation":1},"requests":[{"name":"gpu","devices":[{"driver":"gpu.example.com","pool":"worker-0","name":"gpu-0","attributes":{"model":{"string":"LATEST-GPU-MODEL"},"driverVersion":{"version":"1.0.0"},"type":{"string":"gpu"}}}]}]}`)

	var md metadata.DeviceMetadata
	if err := kubeletplugin.DecodeMetadataFromStream(json.NewDecoder(bytes.NewReader(fileData)), &md); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	fmt.Printf("claim: %s/%s\n", md.Namespace, md.Name)
	fmt.Printf("request: %s, devices: %d\n", md.Requests[0].Name, len(md.Requests[0].Devices))
	fmt.Printf("device: %s (driver: %s, pool: %s)\n", md.Requests[0].Devices[0].Name, md.Requests[0].Devices[0].Driver, md.Requests[0].Devices[0].Pool)
	// Output:
	// claim: default/my-claim
	// request: gpu, devices: 1
	// device: gpu-0 (driver: gpu.example.com, pool: worker-0)
}

// ExampleDecodeMetadataFromStream_multiVersion demonstrates reading a
// metadata file that contains the same data encoded in multiple API versions
// (newest first). The function skips versions it cannot decode and resolves
// the first compatible one into the internal type.
func ExampleDecodeMetadataFromStream_multiVersion() {
	v1alpha2 := `{"apiVersion":"metadata.resource.k8s.io/v1alpha2","kind":"DeviceMetadata","metadata":{"name":"my-claim","namespace":"default","uid":"uid-1234","generation":1},"requests":[{"name":"gpu","devices":[{"driver":"gpu.example.com","pool":"worker-0","name":"gpu-0"}]}]}`
	v1alpha1 := `{"apiVersion":"metadata.resource.k8s.io/v1alpha1","kind":"DeviceMetadata","metadata":{"name":"my-claim","namespace":"default","uid":"uid-1234","generation":1},"requests":[{"name":"gpu","devices":[{"driver":"gpu.example.com","pool":"worker-0","name":"gpu-0","attributes":{"model":{"string":"LATEST-GPU-MODEL"},"type":{"string":"gpu"}}}]}]}`

	// Simulate a JSON stream with two versions separated by newlines.
	stream := strings.Join([]string{v1alpha2, v1alpha1}, "\n")

	var md metadata.DeviceMetadata
	if err := kubeletplugin.DecodeMetadataFromStream(json.NewDecoder(strings.NewReader(stream)), &md); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	fmt.Printf("claim: %s/%s\n", md.Namespace, md.Name)
	fmt.Printf("request: %s, devices: %d\n", md.Requests[0].Name, len(md.Requests[0].Devices))
	fmt.Printf("device: %s (driver: %s, pool: %s)\n", md.Requests[0].Devices[0].Name, md.Requests[0].Devices[0].Driver, md.Requests[0].Devices[0].Pool)
	// Output:
	// claim: default/my-claim
	// request: gpu, devices: 1
	// device: gpu-0 (driver: gpu.example.com, pool: worker-0)
}
