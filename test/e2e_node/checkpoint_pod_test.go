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

import (
	"strings"
	"testing"
)

func TestPodCheckpointSecurity(t *testing.T) {
	const status = `Pid: 1
Uid: 1000 1000 1000 1000
Gid: 2000 2000 2000 2000
Groups: 3000
CapInh: 0000000000000000
CapPrm: 0000000000000000
CapEff: 0000000000000000
CapBnd: 0000000000000000
CapAmb: 0000000000000000
NoNewPrivs: 1
Seccomp: 2
workload-lsm: system_u:system_r:container_t:s0:c1,c2
exec-lsm: system_u:system_r:container_t:s0:c1,c2`
	for _, tc := range []struct {
		name, old, replacement, wantError string
	}{
		{name: "restricted workload"},
		{name: "primary group included", old: "Groups: 3000", replacement: "Groups: 2000 3000"},
		{name: "root credentials", old: "Uid: 1000 1000 1000 1000", replacement: "Uid: 0 0 0 0", wantError: "Uid"},
		{name: "effective privilege", old: "CapEff: 0000000000000000", replacement: "CapEff: 0000000000000001", wantError: "CapEff"},
		{name: "permitted privilege", old: "CapPrm: 0000000000000000", replacement: "CapPrm: 0000000000000001", wantError: "CapPrm"},
		{name: "bounding privilege", old: "CapBnd: 0000000000000000", replacement: "CapBnd: 0000000000000001", wantError: "CapBnd"},
		{name: "ambient privilege", old: "CapAmb: 0000000000000000", replacement: "CapAmb: 0000000000000001", wantError: "CapAmb"},
		{name: "unexpected group", old: "Groups: 3000", replacement: "Groups: 0 3000", wantError: "supplementary groups"},
		{name: "privilege escalation", old: "NoNewPrivs: 1", replacement: "NoNewPrivs: 0", wantError: "NoNewPrivs"},
		{name: "missing seccomp", old: "Seccomp: 2", replacement: "Seccomp: 0", wantError: "Seccomp"},
		{name: "malformed capability", old: "CapBnd: 0000000000000000", replacement: "CapBnd: invalid", wantError: "CapBnd"},
		{name: "wrong workload label", old: "workload-lsm: system_u:system_r:container_t:s0:c1,c2", replacement: "workload-lsm: unconfined", wantError: "LSM label"},
		{name: "kernel without active LSM", old: "system_u:system_r:container_t:s0:c1,c2", replacement: "unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			observed := status
			if tc.old != "" {
				observed = strings.ReplaceAll(observed, tc.old, tc.replacement)
			}
			err := validatePodCheckpointSecurity(observed)
			if tc.wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("security check returned %v, want error mentioning %q", err, tc.wantError)
			}
		})
	}
}

func TestPodCheckpointRuntimeUnsupported(t *testing.T) {
	for _, test := range []struct {
		name    string
		rpc     string
		message string
		want    bool
	}{
		{name: "storage pressure is not unsupported", rpc: "CheckpointPod", message: "checkpoint failed: pod checkpoint storage has insufficient available bytes or inodes"},
		{
			name:    "kubelet reports CheckpointPod missing",
			rpc:     "CheckpointPod",
			message: "checkpoint failed: runtime does not support Pod checkpoints: rpc error: code = Unimplemented desc = unknown method CheckpointPod for service runtime.v1.RuntimeService",
			want:    true,
		},
		{
			name:    "kubelet reports RestorePod missing",
			rpc:     "RestorePod",
			message: "runtime does not support Pod restore: rpc error: code = Unimplemented desc = method RestorePod not implemented",
			want:    true,
		},
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
