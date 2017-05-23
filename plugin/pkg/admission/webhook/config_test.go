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

package webhook

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/admission"
)

func TestConfigNormalization(t *testing.T) {
	defaultRules := []Rule{
		Rule{
			Type: Skip,
		},
	}
	highRetryBackoff := (maxRetryBackoff / time.Millisecond) + (time.Duration(1) * time.Millisecond)
	kubeConfigFile := "/tmp/kube/config"
	lowRetryBackoff := time.Duration(-1)
	normalizedValidRules := []Rule{
		Rule{
			APIGroups:  []string{""},
			FailAction: Allow,
			Namespaces: []string{"my-ns"},
			Operations: []admission.Operation{
				admission.Connect,
				admission.Create,
				admission.Delete,
				admission.Update,
			},
			Resources:     []string{"pods"},
			ResourceNames: []string{"my-name"},
			Type:          Send,
		},
		Rule{
			FailAction: Deny,
			Type:       Skip,
		},
	}
	rawValidRules := []Rule{
		Rule{
			APIGroups:  []string{""},
			FailAction: FailAction("AlLoW"),
			Namespaces: []string{"my-ns"},
			Operations: []admission.Operation{
				admission.Operation("connect"),
				admission.Operation("CREATE"),
				admission.Operation("DeLeTe"),
				admission.Operation("UPdaTE"),
			},
			Resources:     []string{"pods"},
			ResourceNames: []string{"my-name"},
			Type:          RuleType("SenD"),
		},
		Rule{
			FailAction: Deny,
			Type:       Skip,
		},
	}
	unknownFailAction := FailAction("Unknown")
	unknownOperation := admission.Operation("Unknown")
	unknownRuleType := RuleType("Allow")
	tests := []struct {
		test             string
		rawConfig        GenericAdmissionWebhookConfig
		normalizedConfig GenericAdmissionWebhookConfig
		err              error
	}{
		{
			test:      "kubeConfigFile was not provided (error)",
			rawConfig: GenericAdmissionWebhookConfig{},
			err:       fmt.Errorf(errMissingKubeConfigFile),
		},
		{
			test: "retryBackoff was not provided (use default)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules:          defaultRules,
			},
			normalizedConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				RetryBackoff:   defaultRetryBackoff,
				Rules:          defaultRules,
			},
		},
		{
			test: "retryBackoff was below minimum value (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				RetryBackoff:   lowRetryBackoff,
				Rules:          defaultRules,
			},
			err: fmt.Errorf(errRetryBackoffOutOfRange, lowRetryBackoff*time.Millisecond, minRetryBackoff, maxRetryBackoff),
		},
		{
			test: "retryBackoff was above maximum value (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				RetryBackoff:   highRetryBackoff,
				Rules:          defaultRules,
			},
			err: fmt.Errorf(errRetryBackoffOutOfRange, highRetryBackoff*time.Millisecond, minRetryBackoff, maxRetryBackoff),
		},
		{
			test: "rules should have at least one rule (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
			},
			err: fmt.Errorf(errOneRuleRequired),
		},
		{
			test: "fail action was not provided (use default)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules: []Rule{
					Rule{
						Type: Skip,
					},
				},
			},
			normalizedConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				RetryBackoff:   defaultRetryBackoff,
				Rules: []Rule{
					Rule{
						FailAction: defaultFailAction,
						Type:       Skip,
					},
				},
			},
		},
		{
			test: "rule has invalid fail action (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules: []Rule{
					Rule{
						FailAction: unknownFailAction,
					},
				},
			},
			err: fmt.Errorf(errInvalidFailAction, 0, unknownFailAction, Allow, Deny),
		},
		{
			test: "rule has invalid operation (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules: []Rule{
					Rule{
						Operations: []admission.Operation{unknownOperation},
					},
				},
			},
			err: fmt.Errorf(errInvalidRuleOperation, 0, 0, unknownOperation),
		},
		{
			test: "rule has invalid type (error)",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules: []Rule{
					Rule{
						Type: unknownRuleType,
					},
				},
			},
			err: fmt.Errorf(errInvalidRuleType, 0, unknownRuleType, Send, Skip),
		},
		{
			test: "valid configuration",
			rawConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				Rules:          rawValidRules,
			},
			normalizedConfig: GenericAdmissionWebhookConfig{
				KubeConfigFile: kubeConfigFile,
				RetryBackoff:   defaultRetryBackoff,
				Rules:          normalizedValidRules,
			},
			err: nil,
		},
	}

	for _, tt := range tests {
		err := normalizeConfig(&tt.rawConfig)
		if err == nil {
			if tt.err != nil {
				// Ensure that expected errors are produced
				t.Errorf("%s: expected error but did not produce one", tt.test)
			} else if !reflect.DeepEqual(tt.rawConfig, tt.normalizedConfig) {
				// Ensure that valid configurations are structured properly
				t.Errorf("%s: normalized config mismtach. got: %v expected: %v", tt.test, tt.rawConfig, tt.normalizedConfig)
			}
		} else {
			if tt.err == nil {
				// Ensure that unexpected errors are not produced
				t.Errorf("%s: unexpected error: %v", tt.test, err)
			} else if err != nil && tt.err != nil && err.Error() != tt.err.Error() {
				// Ensure that expected errors are formated properly
				t.Errorf("%s: error message mismatch. got: '%v' expected: '%v'", tt.test, err, tt.err)
			}
		}
	}
}
