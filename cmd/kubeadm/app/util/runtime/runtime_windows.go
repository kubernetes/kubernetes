// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/klog/v2"
	"net/url"
	"strings"

	winio "github.com/Microsoft/go-winio"
)

const (
	dockerSocket     = "npipe:////./pipe/docker_engine"         // The Docker socket is not CRI compatible
	containerdSocket = "npipe:////./pipe/containerd-containerd" // Proposed containerd named pipe for Windows
)

// isExistingSocket checks if path exists and is domain socket
func isExistingSocket(path string) bool {
	// url.Parse doesn't recognize \, so replace with / first.
	endpoint := strings.Replace(path, "\\", "/", -1)
	u, err := url.Parse(endpoint)
	if err != nil {
		klog.Warningf("Could not parse the Windows container runtime endpoint: %v", err)
		return false
	}

	if u.Scheme == "" {
		klog.Warningf("Using %q as endpoint is deprecated, please consider using full url format", endpoint)
		return false
	} else if u.Scheme != "tcp" && u.Scheme != "npipe" {
		klog.Warningf("Protocol %q not supported", u.Scheme)
		return false
	}

	var dialPath string
	if strings.HasPrefix(u.Path, "//./pipe") {
		dialPath = u.Path
	} else {
		// fallback host if not provided.
		host := u.Host
		if host == "" {
			host = "."
		}
		dialPath = fmt.Sprintf("//%s%s", host, u.Path)
	}

	_, err = winio.DialPipe(dialPath, nil)
	if err != nil {
		return false
	}

	return true
}
