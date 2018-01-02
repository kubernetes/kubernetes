// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package hypervisor

// KvmHypervisor structure describes KVM hypervisor binary and its parameters
type KvmHypervisor struct {
	Bin          string
	KernelParams []string
}

// InitKernelParams sets debug and common parameters passed to the kernel
func (hv *KvmHypervisor) InitKernelParams(isDebug bool) {
	hv.KernelParams = append(hv.KernelParams, []string{
		"console=hvc0",
		"init=/usr/lib/systemd/systemd",
		"no_timer_check",
		"noreplace-smp",
		"tsc=reliable"}...)

	if isDebug {
		hv.KernelParams = append(hv.KernelParams, []string{
			"debug",
			"systemd.log_level=debug",
			"systemd.show_status=true",
		}...)
	} else {
		hv.KernelParams = append(hv.KernelParams, []string{
			"systemd.show_status=false",
			"systemd.log_target=null",
			"rd.udev.log-priority=3",
			"quiet=vga",
			"quiet systemd.log_level=emerg",
		}...)
	}
}
