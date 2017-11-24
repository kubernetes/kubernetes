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

// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs pkg/nvme/ioctl_types.go

package nvme

type nvmeAdminCmd struct {
	Opcode       uint8
	Flags        uint8
	Rsvd1        uint16
	Nsid         uint32
	Cdw2         uint32
	Cdw3         uint32
	Metadata     uint64
	Addr         uint64
	Metadata_len uint32
	Data_len     uint32
	Cdw10        uint32
	Cdw11        uint32
	Cdw12        uint32
	Cdw13        uint32
	Cdw14        uint32
	Cdw15        uint32
	Timeout_ms   uint32
	Result       uint32
}

const sizeof_nvmeAdminCmd = 0x48
