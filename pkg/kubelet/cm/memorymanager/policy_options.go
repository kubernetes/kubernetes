/*
Copyright The Kubernetes Authors.

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

package memorymanager

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

const (
	MemoryDriftTolerance string = "memory-drift-tolerance"

	memoryDriftToleranceAuto = "auto"
	memoryDriftToleranceOff  = "off"
)

var (
	alphaOptions = sets.New[string](
		MemoryDriftTolerance,
	)
	betaOptions   = sets.New[string]()
	stableOptions = sets.New[string]()
)

func CheckPolicyOptionAvailable(option string) error {
	if !alphaOptions.Has(option) && !betaOptions.Has(option) && !stableOptions.Has(option) {
		return fmt.Errorf("unknown Memory Manager Policy option: %q", option)
	}

	if alphaOptions.Has(option) && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.MemoryManagerDriftTolerance) {
		return fmt.Errorf("memory manager policy option %q requires the %s feature gate", option, kubefeatures.MemoryManagerDriftTolerance)
	}

	return nil
}

type PolicyOptions struct {
	MaxMemoryDrift uint64
}

func NewPolicyOptions(logger klog.Logger, policyOptions map[string]string) (PolicyOptions, error) {
	opts := PolicyOptions{}

	for name, value := range policyOptions {
		if err := CheckPolicyOptionAvailable(name); err != nil {
			return opts, err
		}

		switch name {
		case MemoryDriftTolerance:
			maxMemoryDrift, err := parseMemoryDriftTolerance(logger, value)
			if err != nil {
				return opts, err
			}
			opts.MaxMemoryDrift = maxMemoryDrift
		default:
			return opts, fmt.Errorf("unsupported memorymanager option: %q (%s)", name, value)
		}
	}
	return opts, nil
}

func parseMemoryDriftTolerance(logger klog.Logger, value string) (uint64, error) {
	switch value {
	case memoryDriftToleranceAuto:
		maxMemoryDrift, _ := memoryDriftFromKernelImage(logger, procIomemPath)
		return maxMemoryDrift, nil
	case memoryDriftToleranceOff:
		return 0, nil
	}
	quantity, err := resource.ParseQuantity(value)
	if err != nil {
		return 0, fmt.Errorf("bad value for option %q: %w", MemoryDriftTolerance, err)
	}
	if quantity.Sign() < 0 {
		return 0, fmt.Errorf("bad value for option %q: %q is negative", MemoryDriftTolerance, value)
	}
	return uint64(quantity.Value()), nil
}
