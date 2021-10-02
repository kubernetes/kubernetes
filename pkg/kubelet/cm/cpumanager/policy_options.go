/*
Copyright 2021 The Kubernetes Authors.

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

package cpumanager

import (
	"fmt"
	"strconv"
)

const (
	// FullPCPUsOnlyOption is the name of the CPU Manager policy option
	FullPCPUsOnlyOption string = "full-pcpus-only"
)

type StaticPolicyOptions struct {
	// flag to enable extra allocation restrictions to avoid
	// different containers to possibly end up on the same core.
	// we consider "core" and "physical CPU" synonim here, leaning
	// towards the terminoloy k8s hints. We acknowledge this is confusing.
	//
	// looking at https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/,
	// any possible naming scheme will lead to ambiguity to some extent.
	// We picked "pcpu" because it the established docs hints at vCPU already.
	FullPhysicalCPUsOnly bool
}

func NewStaticPolicyOptions(policyOptions map[string]string) (StaticPolicyOptions, error) {
	opts := StaticPolicyOptions{}
	for name, value := range policyOptions {
		switch name {
		case FullPCPUsOnlyOption:
			optValue, err := strconv.ParseBool(value)
			if err != nil {
				return opts, fmt.Errorf("bad value for option %q: %w", name, err)
			}
			opts.FullPhysicalCPUsOnly = optValue
		default:
			return opts, fmt.Errorf("unsupported cpumanager option: %q (%s)", name, value)
		}
	}
	return opts, nil
}
