/*
Copyright 2016 The Kubernetes Authors.

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

package ipconfig

import (
	"runtime"
	"strings"

	"k8s.io/klog/v2"

	utilexec "k8s.io/utils/exec"
)

// Interface is an injectable interface for running ipconfig commands.  Implementations must be goroutine-safe.
type Interface interface {
	// GetDNSSuffixSearchList returns the list of DNS suffix to search
	GetDNSSuffixSearchList() ([]string, error)
}

const (
	cmdIpconfig string = "ipconfig"

	cmdDefaultArgs string = "/all"

	dnsSuffixSearchLisLabel string = "DNS Suffix Search List"
)

// runner implements Interface in terms of exec("ipconfig").
type runner struct {
	exec utilexec.Interface
}

// New returns a new Interface which will exec ipconfig.
func New(exec utilexec.Interface) Interface {
	runner := &runner{
		exec: exec,
	}
	return runner
}

// GetDNSSuffixSearchList returns the list of DNS suffix to search
func (runner *runner) GetDNSSuffixSearchList() ([]string, error) {
	// Parse the DNS suffix search list from ipconfig output
	// ipconfig /all on Windows displays the entry of DNS suffix search list
	// An example output contains:
	//
	// DNS Suffix Search List. . . . . . : example1.com
	//                                     example2.com
	//
	// TODO: this does not work when the label is localized
	suffixList := []string{}
	if runtime.GOOS != "windows" {
		klog.V(1).Infof("ipconfig not supported on GOOS=%s", runtime.GOOS)
		return suffixList, nil
	}

	out, err := runner.exec.Command(cmdIpconfig, cmdDefaultArgs).Output()

	if err == nil {
		lines := strings.Split(string(out), "\n")
		for i, line := range lines {
			if trimmed := strings.TrimSpace(line); strings.HasPrefix(trimmed, dnsSuffixSearchLisLabel) {
				if parts := strings.Split(trimmed, ":"); len(parts) > 1 {
					if trimmed := strings.TrimSpace(parts[1]); trimmed != "" {
						suffixList = append(suffixList, strings.TrimSpace(parts[1]))
					}
					for j := i + 1; j < len(lines); j++ {
						if trimmed := strings.TrimSpace(lines[j]); trimmed != "" && !strings.Contains(trimmed, ":") {
							suffixList = append(suffixList, trimmed)
						} else {
							break
						}
					}
				}
				break
			}
		}
	} else {
		klog.V(1).Infof("Running %s %s failed: %v", cmdIpconfig, cmdDefaultArgs, err)
	}

	return suffixList, err
}
