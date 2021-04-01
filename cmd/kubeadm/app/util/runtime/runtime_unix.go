// +build !windows

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
	"net"
	"net/url"

	"k8s.io/klog/v2"
)

const (
	dockerSocket     = "unix:///var/run/docker.sock" // The Docker socket is not CRI compatible
	containerdSocket = "unix:///run/containerd/containerd.sock"
)

// isExistingSocket checks if path exists and is domain socket
func isExistingSocket(path string) bool {
	u, err := url.Parse(path)
	if err != nil {
		return false
	}
	// TODO: remove this warning and Scheme override once paths without scheme are not supported
	if u.Scheme != "unix" {
		klog.Warningf("The Unix socket path %q must be prefixed with \"unix://\". "+
			"In future releases this can cause an error on the side of the kubelet. "+
			"Please update your configuration", path)
		u.Scheme = "unix"
	}
	c, err := net.Dial(u.Scheme, u.Path)
	if err != nil {
		return false
	}
	defer c.Close()
	return true
}
