/*
Copyright 2026 The Kubernetes Authors.

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

package dynamicresources

import (
	"bytes"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2/textlogger"
)

// TestRedactedAllocationLogs renders the payload-bearing log entries used for
// successful binding and pending-allocation reuse into a full buffer.
func TestRedactedAllocationLogs(t *testing.T) {
	const canary = "CANARY-10.0.0.0-secret"
	alloc := &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{
		Results: []resourceapi.DeviceRequestAllocationResult{{Request: "req", Driver: "drv", Pool: "pool", Device: "dev-0"}},
		Config: []resourceapi.DeviceAllocationConfiguration{{Source: resourceapi.AllocationConfigSourceClaim,
			DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: "drv", Parameters: runtime.RawExtension{Raw: []byte(`{"subnet":"` + canary + `"}`)}}}}},
	}}
	var buf bytes.Buffer
	logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Verbosity(10), textlogger.Output(&buf)))
	logger.V(5).Info("reserved", "resourceclaim", "ns/claim", "allocation", redactedAllocation{alloc})
	logger.V(5).Info("reusing pending allocation", "resourceclaim", "ns/claim", "allocation", redactedAllocation{alloc})
	out := buf.String()
	if strings.Contains(out, canary) {
		t.Fatalf("opaque parameters leaked into log:\n%s", out)
	}
	for _, want := range []string{"dev-0", "pool", `\u003credacted\u003e`, "ns/claim"} {
		if !strings.Contains(out, want) {
			t.Errorf("log missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(string(alloc.Devices.Config[0].Opaque.Parameters.Raw), canary) {
		t.Error("redaction mutated the original allocation")
	}
}
