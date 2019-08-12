/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"testing"

	conversion "k8s.io/apimachinery/pkg/conversion"
	componentbaseconfig "k8s.io/component-base/config"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	v1alpha1 "k8s.io/kube-scheduler/config/v1alpha1"
	config "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestV1alpha1ToConfigKubeSchedulerLeaderElectionConfiguration(t *testing.T) {
	configuration := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{
		LockObjectName:      "name",
		LockObjectNamespace: "namespace",
		LeaderElectionConfiguration: componentbaseconfigv1alpha1.LeaderElectionConfiguration{
			ResourceName:      "name",
			ResourceNamespace: "namespace",
		},
	}
	emptyLockObjectNameConfig := configuration.DeepCopy()
	emptyLockObjectNameConfig.LockObjectName = ""

	emptyLockObjectNamespaceConfig := configuration.DeepCopy()
	emptyLockObjectNamespaceConfig.LockObjectNamespace = ""

	emptyResourceNameConfig := configuration.DeepCopy()
	emptyResourceNameConfig.ResourceName = ""

	emptyResourceNamespaceConfig := configuration.DeepCopy()
	emptyResourceNamespaceConfig.ResourceNamespace = ""

	differentNameConfig := configuration.DeepCopy()
	differentNameConfig.LockObjectName = "name1"

	differentNamespaceConfig := configuration.DeepCopy()
	differentNamespaceConfig.LockObjectNamespace = "namespace1"

	emptyconfig := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{}

	scenarios := map[string]struct {
		expectedResourceNamespace string
		expectedResourceName      string
		expectedToFailed          bool
		config                    *v1alpha1.KubeSchedulerLeaderElectionConfiguration
	}{
		"both-set-same-name-and-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    configuration,
		},
		"not-set-lock-object-name": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyLockObjectNameConfig,
		},
		"not-set-lock-object-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyLockObjectNamespaceConfig,
		},
		"not-set-resource-name": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyResourceNameConfig,
		},
		"not-set-resource-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyResourceNamespaceConfig,
		},
		"set-different-name": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          true,
			config:                    differentNameConfig,
		},
		"set-different-namespace": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          true,
			config:                    differentNamespaceConfig,
		},
		"set-empty-name-and-namespace": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          false,
			config:                    emptyconfig,
		},
	}
	for name, scenario := range scenarios {
		out := &config.KubeSchedulerLeaderElectionConfiguration{}
		s := conversion.Scope(nil)
		err := Convert_v1alpha1_KubeSchedulerLeaderElectionConfiguration_To_config_KubeSchedulerLeaderElectionConfiguration(scenario.config, out, s)
		if err == nil && scenario.expectedToFailed {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if err == nil && !scenario.expectedToFailed {
			if out.ResourceName != scenario.expectedResourceName {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceName: %s, expectedResourceName: %s", name, out.ResourceName, scenario.expectedResourceName)
			}
			if out.ResourceNamespace != scenario.expectedResourceNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceNamespace: %s, expectedResourceNamespace: %s", name, out.ResourceNamespace, scenario.expectedResourceNamespace)
			}
		}
		if err != nil && !scenario.expectedToFailed {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
		}
	}
}

func TestConfigToV1alpha1KubeSchedulerLeaderElectionConfiguration(t *testing.T) {
	configuration := &config.KubeSchedulerLeaderElectionConfiguration{
		LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
			ResourceName:      "name",
			ResourceNamespace: "namespace",
		},
	}
	emptyconfig := &config.KubeSchedulerLeaderElectionConfiguration{}

	scenarios := map[string]struct {
		expectedResourceNamespace   string
		expectedResourceName        string
		expectedLockObjectNamespace string
		expectedLockObjectName      string
		expectedToFailed            bool
		config                      *config.KubeSchedulerLeaderElectionConfiguration
	}{
		"both-set-name-and-namespace": {
			expectedResourceNamespace:   "namespace",
			expectedResourceName:        "name",
			expectedLockObjectNamespace: "namespace",
			expectedLockObjectName:      "name",
			expectedToFailed:            false,
			config:                      configuration,
		},
		"set-empty-name-and-namespace": {
			expectedResourceNamespace:   "",
			expectedResourceName:        "",
			expectedLockObjectNamespace: "",
			expectedLockObjectName:      "",
			expectedToFailed:            false,
			config:                      emptyconfig,
		},
	}
	for name, scenario := range scenarios {
		out := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{}
		s := conversion.Scope(nil)
		err := Convert_config_KubeSchedulerLeaderElectionConfiguration_To_v1alpha1_KubeSchedulerLeaderElectionConfiguration(scenario.config, out, s)
		if err == nil && scenario.expectedToFailed {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if err == nil && !scenario.expectedToFailed {
			if out.ResourceName != scenario.expectedResourceName {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceName: %s, expectedResourceName: %s", name, out.ResourceName, scenario.expectedResourceName)
			}
			if out.LockObjectName != scenario.expectedLockObjectName {
				t.Errorf("Unexpected success for scenario: %s, out.LockObjectName: %s, expectedLockObjectName: %s", name, out.LockObjectName, scenario.expectedLockObjectName)
			}
			if out.ResourceNamespace != scenario.expectedResourceNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceNamespace: %s, expectedResourceNamespace: %s", name, out.ResourceNamespace, scenario.expectedResourceNamespace)
			}
			if out.LockObjectNamespace != scenario.expectedLockObjectNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.LockObjectNamespace: %s, expectedLockObjectNamespace: %s", name, out.LockObjectNamespace, scenario.expectedLockObjectNamespace)
			}
		}
		if err != nil && !scenario.expectedToFailed {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
		}
	}
}
