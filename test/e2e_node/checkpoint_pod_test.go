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

package e2enode

import "testing"

func TestPodCheckpointRuntimeUnsupported(t *testing.T) {
	for _, test := range []struct {
		name    string
		rpc     string
		message string
		want    bool
	}{
		{
			name:    "checkpoint RPC missing",
			rpc:     "CheckpointPod",
			message: "rpc error: code = Unimplemented desc = unknown method CheckpointPod for service runtime.v1.RuntimeService",
			want:    true,
		},
		{
			name:    "restore RPC missing",
			rpc:     "RestorePod",
			message: "rpc error: code = Unimplemented desc = method RestorePod not implemented",
			want:    true,
		},
		{
			name:    "checkpoint does not hide restore failure",
			rpc:     "CheckpointPod",
			message: "rpc error: code = Unimplemented desc = method RestorePod not implemented",
		},
		{
			name:    "restore does not hide checkpoint failure",
			rpc:     "RestorePod",
			message: "rpc error: code = Unimplemented desc = unknown method CheckpointPod for service runtime.v1.RuntimeService",
		},
		{
			name:    "unrelated RPC unimplemented",
			rpc:     "RestorePod",
			message: "RestorePod failed: rpc error: code = Unimplemented desc = method CreateContainer not implemented",
		},
		{
			name:    "different RPC with matching prefix",
			rpc:     "CheckpointPod",
			message: "rpc error: code = Unimplemented desc = unknown method CheckpointPodStatus for service runtime.v1.RuntimeService",
		},
		{
			name:    "ordinary failure",
			rpc:     "CheckpointPod",
			message: "rpc error: code = Internal desc = failed to write checkpoint: no space left on device",
		},
		{
			name:    "Unimplemented in a diagnostic",
			rpc:     "RestorePod",
			message: "restore failed in UnimplementedDriver: invalid checkpoint",
		},
		{
			name:    "CRIU dependency unavailable",
			rpc:     "CheckpointPod",
			message: "rpc error: code = Unknown desc = CRIU binary not found or too old (<31600)",
			want:    true,
		},
		{
			name:    "CRIU dependency disabled on restore",
			rpc:     "RestorePod",
			message: "rpc error: code = Unknown desc = criu support is disabled by configuration",
			want:    true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := podCheckpointRuntimeUnsupported(test.rpc, test.message); got != test.want {
				t.Errorf("unsupported %s runtime = %v, want %v for %q", test.rpc, got, test.want, test.message)
			}
		})
	}
}
