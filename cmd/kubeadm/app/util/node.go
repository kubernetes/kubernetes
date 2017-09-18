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
	systemdutil "github.com/coreos/go-systemd/util"

	"k8s.io/kubernetes/pkg/util/node"
)

// GetDefaultNodeName returns a suitable default NodeName
func GetDefaultNodeName(override string) (string, error) {
	if override != "" {
		return override, nil
	}

	if mid, err := systemdutil.GetMachineID(); err != nil {
		return "node-" + mid, nil
	}

	return node.GetHostname(""), nil
}
