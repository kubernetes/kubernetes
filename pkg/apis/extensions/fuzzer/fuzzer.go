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
	"fmt"

	fuzz "github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// Funcs returns the fuzzer functions for the extensions api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(j *extensions.DeploymentSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			pds := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
			j.ProgressDeadlineSeconds = &pds
		},
		func(j *extensions.DeploymentStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []extensions.DeploymentStrategyType{extensions.RecreateDeploymentStrategyType, extensions.RollingUpdateDeploymentStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != extensions.RollingUpdateDeploymentStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := extensions.RollingUpdateDeployment{}
				if c.RandBool() {
					rollingUpdate.MaxUnavailable = intstr.FromInt(int(c.Rand.Int31()))
					rollingUpdate.MaxSurge = intstr.FromInt(int(c.Rand.Int31()))
				} else {
					rollingUpdate.MaxSurge = intstr.FromString(fmt.Sprintf("%d%%", c.Rand.Int31()))
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
		func(psp *extensions.PodSecurityPolicySpec, c fuzz.Continue) {
			c.FuzzNoCustom(psp) // fuzz self without calling this function again
			runAsUserRules := []extensions.RunAsUserStrategy{extensions.RunAsUserStrategyMustRunAsNonRoot, extensions.RunAsUserStrategyMustRunAs, extensions.RunAsUserStrategyRunAsAny}
			psp.RunAsUser.Rule = runAsUserRules[c.Rand.Intn(len(runAsUserRules))]
			seLinuxRules := []extensions.SELinuxStrategy{extensions.SELinuxStrategyRunAsAny, extensions.SELinuxStrategyMustRunAs}
			psp.SELinux.Rule = seLinuxRules[c.Rand.Intn(len(seLinuxRules))]
		},
		func(s *extensions.Scale, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			// TODO: Implement a fuzzer to generate valid keys, values and operators for
			// selector requirements.
			if s.Status.Selector != nil {
				s.Status.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"testlabelkey": "testlabelval",
					},
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "testkey",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"val1", "val2", "val3"},
						},
					},
				}
			}
		},
		func(j *extensions.DaemonSetSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
		},
		func(j *extensions.DaemonSetUpdateStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []extensions.DaemonSetUpdateStrategyType{extensions.RollingUpdateDaemonSetStrategyType, extensions.OnDeleteDaemonSetStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != extensions.RollingUpdateDaemonSetStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := extensions.RollingUpdateDaemonSet{}
				if c.RandBool() {
					if c.RandBool() {
						rollingUpdate.MaxUnavailable = intstr.FromInt(1 + int(c.Rand.Int31()))
					} else {
						rollingUpdate.MaxUnavailable = intstr.FromString(fmt.Sprintf("%d%%", 1+c.Rand.Int31()))
					}
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
	}
}
