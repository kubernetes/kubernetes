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

package hvqemu

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coreos/rkt/stage1/init/kvm"
	"github.com/coreos/rkt/stage1/init/kvm/hypervisor"
)

// StartCmd takes path to stage1, name of the machine, path to kernel, network describers, memory in megabytes
// and quantity of cpus and prepares command line to run QEMU process
func StartCmd(wdPath, name, kernelPath string, nds []kvm.NetDescriber, cpu, mem int64, debug bool) []string {
	var (
		driverConfiguration = hypervisor.KvmHypervisor{
			Bin: "./qemu",
			KernelParams: []string{
				"root=/dev/root",
				"rootfstype=9p",
				"rootflags=trans=virtio,version=9p2000.L,cache=mmap",
				"rw",
				"systemd.default_standard_error=journal+console",
				"systemd.default_standard_output=journal+console",
			},
		}
	)

	driverConfiguration.InitKernelParams(debug)

	cmd := []string{
		filepath.Join(wdPath, driverConfiguration.Bin),
		"-L", wdPath,
		"-no-reboot",
		"-display", "none",
		"-enable-kvm",
		"-smp", strconv.FormatInt(cpu, 10),
		"-m", strconv.FormatInt(mem, 10),
		"-kernel", kernelPath,
		"-fsdev", "local,id=root,path=stage1/rootfs,security_model=none",
		"-device", "virtio-9p-pci,fsdev=root,mount_tag=/dev/root",
		"-append", fmt.Sprintf("%s", strings.Join(driverConfiguration.KernelParams, " ")),
		"-chardev", "stdio,id=virtiocon0,signal=off",
		"-device", "virtio-serial",
		"-device", "virtconsole,chardev=virtiocon0",
	}
	return append(cmd, kvmNetArgs(nds)...)
}

// kvmNetArgs returns additional arguments that need to be passed
// to qemu to configure networks properly. Logic is based on
// network configuration extracted from Networking struct
// and essentially from activeNets that expose NetDescriber behavior
func kvmNetArgs(nds []kvm.NetDescriber) []string {
	var qemuArgs []string

	for _, nd := range nds {
		qemuArgs = append(qemuArgs, []string{"-net", "nic,model=virtio"}...)
		qemuNic := fmt.Sprintf("tap,ifname=%s,script=no,downscript=no,vhost=on", nd.IfName())
		qemuArgs = append(qemuArgs, []string{"-net", qemuNic}...)
	}

	return qemuArgs
}
