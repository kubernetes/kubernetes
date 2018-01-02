// Copyright 2015 The rkt Authors
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

package tuntap

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/hashicorp/errwrap"
)

const (
	Tun = 0x1
	Tap = 0x2

	NoPi     = 0x1000
	OneQueue = 0x2000
	VnetHdr  = 0x4000
	TunExcl  = 0x8000
)

type ifReq struct {
	Name  [0x10]byte
	Flags uint16
	pad   [0x28 - 0x12]byte
}

type Interface struct {
	name string
	file *os.File
}

func Open(ifName string, kind uint16) (*Interface, error) {
	if ifName == "" {
		ifName = "tap%d"
	}
	file, err := os.OpenFile("/dev/net/tun", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}

	var req ifReq
	copy(req.Name[:15], ifName)
	req.Flags = kind | NoPi | VnetHdr

	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, file.Fd(), uintptr(syscall.TUNSETIFF), uintptr(unsafe.Pointer(&req)))
	if errno != 0 {
		file.Close()
		return nil, fmt.Errorf("ioctl failed (TUNSETIFF): %v", errno)
	}

	return &Interface{
		name: string(req.Name[:bytes.IndexByte(req.Name[:], 0)]),
		file: file,
	}, nil
}

func (iface *Interface) Close() error {
	return iface.file.Close()
}

func operateOnIface(name string, kind uint16, persistency uintptr) (string, error) {
	iface, err := Open(name, kind)
	if err != nil {
		return "", err
	}

	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, iface.file.Fd(), uintptr(syscall.TUNSETPERSIST), persistency)
	err = iface.Close()
	if err != nil {
		return "", errwrap.Wrap(errors.New("iface close error "), err)
	}

	if errno != 0 {
		return "", fmt.Errorf("ioctl failed (TUNSETPERSIST): %v", errno)
	}

	return iface.name, nil
}

func CreatePersistentIface(nameTemplate string, kind uint16) (string, error) {
	return operateOnIface(nameTemplate, kind, 1)
}

func RemovePersistentIface(name string, kind uint16) error {
	_, err := operateOnIface(name, kind, 0)
	return err
}
