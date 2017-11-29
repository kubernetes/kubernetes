// +build linux,amd64

/*
Copyright 2017 The Kubernetes Authors.

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

package nvme

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"unsafe"
)

const (
	cmd_NVME_ID_CNS_NS   = 0x00
	cmd_NVME_ID_CNS_CTRL = 0x01
)

const nvme_admin_identify = 0x06
const (
	cmd_NVME_IOCTL_ID        = uintptr(0x40)
	cmd_NVME_IOCTL_ADMIN_CMD = uintptr(0x41)
)

const (
	ioctlmode_IOC_NONE  = 0
	ioctlmode_IOC_READ  = 2
	ioctlmode_IOC_WRITE = 1
)

// Identify returns the nvme volume identifier for an nvme device
// Based on the calls made by "nvme list"
func Identify(p string) (string, bool, error) {
	fd, err := os.Open(p)
	if err != nil {
		return "", false, err
	}
	defer fd.Close()

	if _, err := nvmeIdentifyCtrl(fd.Fd()); err != nil {
		return "", false, err
	}
	nsid, err := nvmeGetNsid(fd.Fd())
	if err != nil {
		return "", false, err
	}
	info, err := nvmeIdentifyNs(fd.Fd(), nsid)
	if err != nil {
		return "", false, err
	}

	return info, true, nil
}

func nvmeIdentify(fd uintptr, nsid uint32, cdw10 uint32) (string, error) {
	dataLength := 0x1000
	var buf [0x1000]byte
	cmd := nvmeAdminCmd{
		Opcode:   nvme_admin_identify,
		Nsid:     nsid,
		Addr:     uint64(uintptr(unsafe.Pointer(&buf[0]))),
		Data_len: uint32(dataLength),
		Cdw10:    cdw10,
	}

	code := (uintptr(ioctlmode_IOC_READ|ioctlmode_IOC_WRITE) << 30) | cmd_NVME_IOCTL_ADMIN_CMD | (uintptr('N') << 8) | (uintptr(sizeof_nvmeAdminCmd) << 16)

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, code, uintptr(unsafe.Pointer(&cmd))); err != 0 {
		return "", fmt.Errorf("error from nvme_admin_identify syscall: %v", err)
	}
	info := string(buf[4:24])
	info = strings.Trim(info, " \x00")
	return info, nil
}

func nvmeGetNsid(fd uintptr) (uint32, error) {
	ns := uint32(0)

	code := (uintptr(ioctlmode_IOC_NONE) << 30) | cmd_NVME_IOCTL_ID | (uintptr('N') << 8)

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, code, uintptr(unsafe.Pointer(&ns))); err != 0 {
		return 0, fmt.Errorf("error from NVME_IOCTL_ID syscall: %v", err)
	}
	return ns, nil
}

func nvmeIdentifyCtrl(fd uintptr) (string, error) {
	return nvmeIdentify(fd, 0, cmd_NVME_ID_CNS_CTRL)
}

func nvmeIdentifyNs(fd uintptr, nsid uint32) (string, error) {
	return nvmeIdentify(fd, nsid, cmd_NVME_ID_CNS_NS)
}
