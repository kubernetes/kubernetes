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

package fuzzer

import (
	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/policy"
)

// Funcs returns the fuzzer functions for the policy api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *policy.PodDisruptionBudgetStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.PodDisruptionsAllowed = int32(c.Rand.Intn(2))
		},
		func(psp *policy.PodSecurityPolicySpec, c fuzz.Continue) {
			c.FuzzNoCustom(psp) // fuzz self without calling this function again

			runAsUserRules := []policy.RunAsUserStrategy{
				policy.RunAsUserStrategyMustRunAsNonRoot,
				policy.RunAsUserStrategyMustRunAs,
				policy.RunAsUserStrategyRunAsAny,
			}
			psp.RunAsUser.Rule = runAsUserRules[c.Rand.Intn(len(runAsUserRules))]

			seLinuxRules := []policy.SELinuxStrategy{
				policy.SELinuxStrategyMustRunAs,
				policy.SELinuxStrategyRunAsAny,
			}
			psp.SELinux.Rule = seLinuxRules[c.Rand.Intn(len(seLinuxRules))]

			supplementalGroupsRules := []policy.SupplementalGroupsStrategyType{
				policy.SupplementalGroupsStrategyRunAsAny,
				policy.SupplementalGroupsStrategyMustRunAs,
			}
			psp.SupplementalGroups.Rule = supplementalGroupsRules[c.Rand.Intn(len(supplementalGroupsRules))]

			fsGroupRules := []policy.FSGroupStrategyType{
				policy.FSGroupStrategyMustRunAs,
				policy.FSGroupStrategyRunAsAny,
			}
			psp.FSGroup.Rule = fsGroupRules[c.Rand.Intn(len(fsGroupRules))]
		},
	}
}
