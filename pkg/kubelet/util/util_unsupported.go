// +build !freebsd,!linux,!windows,!darwin

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

package util

import (
	"fmt"
	"net"
	"time"
)

func CreateListener(endpoint string) (net.Listener, error) {
	return nil, fmt.Errorf("CreateListener is unsupported in this build")
}

func GetAddressAndDialer(endpoint string) (string, func(addr string, timeout time.Duration) (net.Conn, error), error) {
	return "", nil, fmt.Errorf("GetAddressAndDialer is unsupported in this build")
}

// LockAndCheckSubPath empty implementation
func LockAndCheckSubPath(volumePath, subPath string) ([]uintptr, error) {
	return []uintptr{}, nil
}

// UnlockPath empty implementation
func UnlockPath(fileHandles []uintptr) {
}
