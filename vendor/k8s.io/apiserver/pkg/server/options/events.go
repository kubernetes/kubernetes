/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"os"
)

// ProcessInfo holds the apiserver process information used to send events
type ProcessInfo struct {
	// Name of the api process to identify events
	Name string

	// Namespace of the api process to send events
	Namespace string
}

// NewProcessInfo returns a new process info with the hostname concatenated to the name given
func NewProcessInfo(name, namespace string) *ProcessInfo {
	// try to concat the hostname if available
	host, _ := os.Hostname()
	if host != "" {
		name = fmt.Sprintf("%s-%s", name, host)
	}
	return &ProcessInfo{
		Name:      name,
		Namespace: namespace,
	}
}

// validateProcessInfo checks for a complete process info
func validateProcessInfo(p *ProcessInfo) error {
	if p == nil {
		return fmt.Errorf("ProcessInfo must be set")
	} else if p.Name == "" {
		return fmt.Errorf("ProcessInfo name must be set")
	} else if p.Namespace == "" {
		return fmt.Errorf("ProcessInfo namespace must be set")
	}
	return nil
}
