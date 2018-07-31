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

package validation

import (
	"fmt"
	"net"

	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func ValidateKubeSchedulerConfiguration(ksc *componentconfig.KubeSchedulerConfiguration) []error {
	allErrors := []error{}

	if utilvalidation.IsInRange(int(ksc.HardPodAffinitySymmetricWeight), 0, 100) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: HardPodAffinitySymmetricWeight %d should be in the range of 0-100", ksc.HardPodAffinitySymmetricWeight))
	}

	if _, _, err := net.SplitHostPort(ksc.HealthzBindAddress); err != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: HealthzBindAddress %s should be the <IP-address>:<port>. Unexpected error %v", ksc.HealthzBindAddress, err))
	}

	if _, _, err := net.SplitHostPort(ksc.MetricsBindAddress); err != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: MetricsBindAddress %s should be the <IP-address>:<port>. Unexpected error %v", ksc.MetricsBindAddress, err))
	}

	return allErrors
}
