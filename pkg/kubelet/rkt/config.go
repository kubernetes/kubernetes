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

package rkt

import "fmt"

// Config stores the global configuration for the rkt runtime.
// Run 'rkt' for more details.
type Config struct {
	// The debug flag for rkt.
	Debug bool
	// The rkt data directory.
	Dir string
	// This flag controls whether we skip image or key verification.
	InsecureSkipVerify bool
	// The local config directory.
	LocalConfigDir string
}

// buildGlobalOptions returns an array of global command line options.
func (c *Config) buildGlobalOptions() []string {
	var result []string
	if c == nil {
		return result
	}

	result = append(result, fmt.Sprintf("--debug=%v", c.Debug))
	result = append(result, fmt.Sprintf("--insecure-skip-verify=%v", c.InsecureSkipVerify))
	if c.LocalConfigDir != "" {
		result = append(result, fmt.Sprintf("--local-config=%s", c.LocalConfigDir))
	}
	if c.Dir != "" {
		result = append(result, fmt.Sprintf("--dir=%s", c.Dir))
	}
	return result
}
