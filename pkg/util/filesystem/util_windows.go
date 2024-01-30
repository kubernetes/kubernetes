//go:build windows
// +build windows

/*
Copyright 2023 The Kubernetes Authors.

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

package filesystem

import (
	"fmt"
	"net"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const (
	// Amount of time to wait between attempting to use a Unix domain socket.
	// As detailed in https://github.com/kubernetes/kubernetes/issues/104584
	// the first attempt will most likely fail, hence the need to retry
	socketDialRetryPeriod = 1 * time.Second
	// Overall timeout value to dial a Unix domain socket, including retries
	socketDialTimeout = 4 * time.Second
)

// IsUnixDomainSocket returns whether a given file is a AF_UNIX socket file
// Note that due to the retry logic inside, it could take up to 4 seconds
// to determine whether or not the file path supplied is a Unix domain socket
func IsUnixDomainSocket(filePath string) (bool, error) {
	// Due to the absence of golang support for os.ModeSocket in Windows (https://github.com/golang/go/issues/33357)
	// we need to dial the file and check if we receive an error to determine if a file is Unix Domain Socket file.

	// Note that querrying for the Reparse Points (https://docs.microsoft.com/en-us/windows/win32/fileio/reparse-points)
	// for the file (using FSCTL_GET_REPARSE_POINT) and checking for reparse tag: reparseTagSocket
	// does NOT work in 1809 if the socket file is created within a bind mounted directory by a container
	// and the FSCTL is issued in the host by the kubelet.

	// If the file does not exist, it cannot be a Unix domain socket.
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return false, fmt.Errorf("File %s not found. Err: %v", filePath, err)
	}

	klog.V(6).InfoS("Function IsUnixDomainSocket starts", "filePath", filePath)
	// As detailed in https://github.com/kubernetes/kubernetes/issues/104584 we cannot rely
	// on the Unix Domain socket working on the very first try, hence the potential need to
	// dial multiple times
	var lastSocketErr error
	err := wait.PollImmediate(socketDialRetryPeriod, socketDialTimeout,
		func() (bool, error) {
			klog.V(6).InfoS("Dialing the socket", "filePath", filePath)
			var c net.Conn
			c, lastSocketErr = net.Dial("unix", filePath)
			if lastSocketErr == nil {
				c.Close()
				klog.V(6).InfoS("Socket dialed successfully", "filePath", filePath)
				return true, nil
			}
			klog.V(6).InfoS("Failed the current attempt to dial the socket, so pausing before retry",
				"filePath", filePath, "err", lastSocketErr, "socketDialRetryPeriod",
				socketDialRetryPeriod)
			return false, nil
		})

	// PollImmediate will return "timed out waiting for the condition" if the function it
	// invokes never returns true
	if err != nil {
		klog.V(2).InfoS("Failed all attempts to dial the socket so marking it as a non-Unix Domain socket. Last socket error along with the error from PollImmediate follow",
			"filePath", filePath, "lastSocketErr", lastSocketErr, "err", err)
		return false, nil
	}
	return true, nil
}
