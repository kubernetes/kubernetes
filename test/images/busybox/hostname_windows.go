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

package main

import (
	"fmt"
	"strings"
	"syscall"
	"unsafe"
)

const aiFQDN = 0x00020000

func getFQDN() (string, error) {
	hostname, err := getHostname()
	if err != nil {
		return "", err
	}
	return lookupCanonicalName(hostname)
}

func lookupCanonicalName(hostname string) (string, error) {
	hostname16, err := syscall.UTF16PtrFromString(hostname)
	if err != nil {
		return "", err
	}

	hints := syscall.AddrinfoW{
		Flags:    aiFQDN,
		Family:   syscall.AF_UNSPEC,
		Socktype: syscall.SOCK_STREAM,
		Protocol: syscall.IPPROTO_IP,
	}
	var result *syscall.AddrinfoW
	if err := syscall.GetAddrInfoW(hostname16, nil, &hints, &result); err != nil {
		return "", fmt.Errorf("resolve hostname %q: %w", hostname, err)
	}
	defer syscall.FreeAddrInfoW(result)

	for addr := result; addr != nil; addr = addr.Next {
		if addr.Canonname == nil {
			continue
		}
		canonical := utf16PtrToString(addr.Canonname)
		if canonical != "" {
			return strings.TrimSuffix(canonical, "."), nil
		}
	}
	return hostname, nil
}

func utf16PtrToString(p *uint16) string {
	var s []uint16
	for ; *p != 0; p = (*uint16)(unsafe.Add(unsafe.Pointer(p), unsafe.Sizeof(*p))) {
		s = append(s, *p)
	}
	return syscall.UTF16ToString(s)
}
