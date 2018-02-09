// +build windows

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

package winstats

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// getCurrentVersionVal gets value of specified key from registry.
func getCurrentVersionVal(key string) (string, error) {
	var h windows.Handle
	if err := windows.RegOpenKeyEx(windows.HKEY_LOCAL_MACHINE,
		windows.StringToUTF16Ptr(`SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\`),
		0,
		windows.KEY_READ,
		&h); err != nil {
		return "", err
	}
	defer windows.RegCloseKey(h)

	var buf [128]uint16
	var typ uint32
	n := uint32(len(buf) * int(unsafe.Sizeof(buf[0]))) // api expects array of bytes, not uint16
	if err := windows.RegQueryValueEx(h,
		windows.StringToUTF16Ptr(key),
		nil,
		&typ,
		(*byte)(unsafe.Pointer(&buf[0])),
		&n); err != nil {
		return "", err
	}

	return windows.UTF16ToString(buf[:]), nil
}

// getVersionRevision gets revision from UBR registry.
func getVersionRevision() (uint16, error) {
	revisionString, err := getCurrentVersionVal("UBR")
	if err != nil {
		return 0, err
	}

	revision, err := windows.UTF16FromString(revisionString)
	if err != nil {
		return 0, err
	}

	return revision[0], nil
}

// getKernelVersion gets the version of windows kernel.
func getKernelVersion() (string, error) {
	// Get CurrentBuildNumber.
	buildNumber, err := getCurrentVersionVal("CurrentBuildNumber")
	if err != nil {
		return "", err
	}

	// Get CurrentMajorVersionNumber.
	majorVersionNumberString, err := getCurrentVersionVal("CurrentMajorVersionNumber")
	if err != nil {
		return "", err
	}
	majorVersionNumber, err := windows.UTF16FromString(majorVersionNumberString)
	if err != nil {
		return "", err
	}

	// Get CurrentMinorVersionNumber.
	minorVersionNumberString, err := getCurrentVersionVal("CurrentMinorVersionNumber")
	if err != nil {
		return "", err
	}
	minorVersionNumber, err := windows.UTF16FromString(minorVersionNumberString)
	if err != nil {
		return "", err
	}

	// Get UBR.
	revision, err := getVersionRevision()
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%d.%d.%s.%d\n", majorVersionNumber[0], minorVersionNumber[0], buildNumber, revision), nil
}

// getOSImageVersion gets the osImage name and version.
func getOSImageVersion() (string, error) {
	productName, err := getCurrentVersionVal("ProductName")
	if err != nil {
		return "", nil
	}

	return productName, nil
}
