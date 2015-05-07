/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"os/exec"
	"strings"

	"github.com/golang/glog"
)

func GetHostname(hostnameOverride string) string {
	hostname := hostnameOverride
	if string(hostname) == "" {
		nodename, err := exec.Command("uname", "-n").Output()
		if err != nil {
			glog.Fatalf("Couldn't determine hostname: %v", err)
		}
		chunks := strings.Split(string(nodename), ".")
		// nodename could be a fully-qualified domain name or not. Take the first
		// word of nodename as the hostname for consistency.
		hostname = chunks[0]
	}
	return strings.ToLower(strings.TrimSpace(hostname))
}
