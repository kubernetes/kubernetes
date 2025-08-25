/*
Copyright 2022 The Kubernetes Authors.

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

package fuzzer

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/resource"
	"sigs.k8s.io/randfill"
)

// Funcs contains the fuzzer functions for the resource group.
//
// Leaving fields empty which then get replaced by the default
// leads to errors during roundtrip tests.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(r *resource.ExactDeviceRequest, c randfill.Continue) {
			c.FillNoCustom(r) // fuzz self without calling this function again

			if r.AllocationMode == "" {
				r.AllocationMode = []resource.DeviceAllocationMode{
					resource.DeviceAllocationModeAll,
					resource.DeviceAllocationModeExactCount,
				}[c.Int31n(2)]
			}
		},
		func(r *resource.DeviceSubRequest, c randfill.Continue) {
			c.FillNoCustom(r) // fuzz self without calling this function again

			if r.AllocationMode == "" {
				r.AllocationMode = []resource.DeviceAllocationMode{
					resource.DeviceAllocationModeAll,
					resource.DeviceAllocationModeExactCount,
				}[c.Int31n(2)]
			}
		},
		func(r *resource.DeviceAllocationConfiguration, c randfill.Continue) {
			c.FillNoCustom(r)
			if r.Source == "" {
				r.Source = []resource.AllocationConfigSource{
					resource.AllocationConfigSourceClass,
					resource.AllocationConfigSourceClaim,
				}[c.Int31n(2)]
			}
		},
		func(r *resource.DeviceToleration, c randfill.Continue) {
			c.FillNoCustom(r)
			if r.Operator == "" {
				r.Operator = []resource.DeviceTolerationOperator{
					resource.DeviceTolerationOpEqual,
					resource.DeviceTolerationOpExists,
				}[c.Int31n(2)]
			}
		},
		func(r *resource.DeviceTaint, c randfill.Continue) {
			c.FillNoCustom(r)
			if r.TimeAdded == nil {
				// Current time is more or less random.
				// Truncate to seconds because sub-second resolution
				// does not survive round-tripping.
				r.TimeAdded = &metav1.Time{Time: time.Now().Truncate(time.Second)}
			}
		},
		func(r *resource.OpaqueDeviceConfiguration, c randfill.Continue) {
			c.FillNoCustom(r)
			// Match the fuzzer default content for runtime.Object.
			//
			// This is necessary because randomly generated content
			// might be valid JSON which changes during re-encoding.
			r.Parameters = runtime.RawExtension{Raw: []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`)}
		},
		func(r *resource.AllocatedDeviceStatus, c randfill.Continue) {
			c.FillNoCustom(r)
			// Match the fuzzer default content for runtime.Object.
			//
			// This is necessary because randomly generated content
			// might be valid JSON which changes during re-encoding.
			r.Data = &runtime.RawExtension{Raw: []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`)}
		},
		func(r *resource.ResourceSliceSpec, c randfill.Continue) {
			c.FillNoCustom(r)
			// Setting AllNodes to false is not allowed. It must be
			// either true or nil.
			if r.AllNodes != nil && !*r.AllNodes {
				r.AllNodes = nil
			}
			if r.NodeName != nil && *r.NodeName == "" {
				r.NodeName = nil
			}
		},
	}
}
