//go:build windows
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

package util

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/Microsoft/go-winio"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const (
	tcpProtocol   = "tcp"
	npipeProtocol = "npipe"
	// Amount of time to wait between attempting to use a Unix domain socket.
	// As detailed in https://github.com/kubernetes/kubernetes/issues/104584
	// the first attempt will most likely fail, hence the need to retry
	socketDialRetryPeriod = 1 * time.Second
	// Overall timeout value to dial a Unix domain socket, including retries
	socketDialTimeout = 4 * time.Second
)

// CreateListener creates a listener on the specified endpoint.
func CreateListener(endpoint string) (net.Listener, error) {
	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	switch protocol {
	case tcpProtocol:
		return net.Listen(tcpProtocol, addr)

	case npipeProtocol:
		return winio.ListenPipe(addr, nil)

	default:
		return nil, fmt.Errorf("only support tcp and npipe endpoint")
	}
}

// GetAddressAndDialer returns the address parsed from the given endpoint and a context dialer.
func GetAddressAndDialer(endpoint string) (string, func(ctx context.Context, addr string) (net.Conn, error), error) {
	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return "", nil, err
	}

	if protocol == tcpProtocol {
		return addr, tcpDial, nil
	}

	if protocol == npipeProtocol {
		return addr, npipeDial, nil
	}

	return "", nil, fmt.Errorf("only support tcp and npipe endpoint")
}

func tcpDial(ctx context.Context, addr string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, tcpProtocol, addr)
}

func npipeDial(ctx context.Context, addr string) (net.Conn, error) {
	return winio.DialPipeContext(ctx, addr)
}

func parseEndpoint(endpoint string) (string, string, error) {
	// url.Parse doesn't recognize \, so replace with / first.
	endpoint = strings.Replace(endpoint, "\\", "/", -1)
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", err
	}

	if u.Scheme == "tcp" {
		return "tcp", u.Host, nil
	} else if u.Scheme == "npipe" {
		if strings.HasPrefix(u.Path, "//./pipe") {
			return "npipe", u.Path, nil
		}

		// fallback host if not provided.
		host := u.Host
		if host == "" {
			host = "."
		}
		return "npipe", fmt.Sprintf("//%s%s", host, u.Path), nil
	} else if u.Scheme == "" {
		return "", "", fmt.Errorf("Using %q as endpoint is deprecated, please consider using full url format", endpoint)
	} else {
		return u.Scheme, "", fmt.Errorf("protocol %q not supported", u.Scheme)
	}
}

// LocalEndpoint empty implementation
func LocalEndpoint(path, file string) (string, error) {
	return "", fmt.Errorf("LocalEndpoints are unsupported in this build")
}

var tickCount = syscall.NewLazyDLL("kernel32.dll").NewProc("GetTickCount64")

// GetBootTime returns the time at which the machine was started, truncated to the nearest second
func GetBootTime() (time.Time, error) {
	currentTime := time.Now()
	output, _, err := tickCount.Call()
	if errno, ok := err.(syscall.Errno); !ok || errno != 0 {
		return time.Time{}, err
	}
	return currentTime.Add(-time.Duration(output) * time.Millisecond).Truncate(time.Second), nil
}

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

// NormalizePath converts FS paths returned by certain go frameworks (like fsnotify)
// to native Windows paths that can be passed to Windows specific code
func NormalizePath(path string) string {
	path = strings.ReplaceAll(path, "/", "\\")
	if strings.HasPrefix(path, "\\") {
		path = "c:" + path
	}
	return path
}
