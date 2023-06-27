/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"testing"

	kubeapiserveradmission "k8s.io/apiserver/pkg/admission"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	basemetrics "k8s.io/component-base/metrics"

	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
)

func TestValidateAPIPriorityAndFairness(t *testing.T) {
	const conflict = "conflicts with --enable-priority-and-fairness=true and --feature-gates=APIPriorityAndFairness=true"
	tests := []struct {
		runtimeConfig    string
		errShouldContain string
	}{
		{
			runtimeConfig:    "api/all=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "api/beta=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta1=false",
			errShouldContain: "",
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta2=false",
			errShouldContain: "",
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta3=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta3=true",
			errShouldContain: "",
		},
	}

	for _, test := range tests {
		t.Run(test.runtimeConfig, func(t *testing.T) {
			options := &Options{
				GenericServerRunOptions: &genericoptions.ServerRunOptions{
					EnablePriorityAndFairness: true,
				},
				APIEnablement: genericoptions.NewAPIEnablementOptions(),
			}
			options.APIEnablement.RuntimeConfig.Set(test.runtimeConfig)

			var errMessageGot string
			if errs := validateAPIPriorityAndFairness(options); len(errs) > 0 {
				errMessageGot = errs[0].Error()
			}
			if !strings.Contains(errMessageGot, test.errShouldContain) {
				t.Errorf("Expected error message to contain: %q, but got: %q", test.errShouldContain, errMessageGot)
			}
		})
	}
}

func TestValidateOptions(t *testing.T) {
	testCases := []struct {
		name         string
		options      *Options
		expectErrors bool
	}{
		{
			name:         "validate master count equal 0",
			expectErrors: true,
			options: &Options{
				GenericServerRunOptions: &genericoptions.ServerRunOptions{},
				Etcd:                    &genericoptions.EtcdOptions{},
				SecureServing:           &genericoptions.SecureServingOptionsWithLoopback{},
				Audit:                   &genericoptions.AuditOptions{},
				Admission: &kubeoptions.AdmissionOptions{
					GenericAdmission: &genericoptions.AdmissionOptions{
						EnablePlugins: []string{"foo"},
						Plugins:       kubeapiserveradmission.NewPlugins(),
					},
					PluginNames: []string{"foo"},
				},
				Authentication: &kubeoptions.BuiltInAuthenticationOptions{
					APIAudiences: []string{"bar"},
					ServiceAccounts: &kubeoptions.ServiceAccountAuthenticationOptions{
						Issuers: []string{"baz"},
					},
				},
				APIEnablement:                genericoptions.NewAPIEnablementOptions(),
				Metrics:                      &basemetrics.Options{},
				ServiceAccountSigningKeyFile: "",
			},
		},
		{
			name:         "validate token request enable not attempted",
			expectErrors: true,
			options: &Options{
				GenericServerRunOptions: &genericoptions.ServerRunOptions{},
				Etcd:                    &genericoptions.EtcdOptions{},
				SecureServing:           &genericoptions.SecureServingOptionsWithLoopback{},
				Audit:                   &genericoptions.AuditOptions{},
				Admission: &kubeoptions.AdmissionOptions{
					GenericAdmission: &genericoptions.AdmissionOptions{
						EnablePlugins: []string{""},
						Plugins:       kubeapiserveradmission.NewPlugins(),
					},
					PluginNames: []string{""},
				},
				Authentication: &kubeoptions.BuiltInAuthenticationOptions{
					ServiceAccounts: &kubeoptions.ServiceAccountAuthenticationOptions{},
				},
				APIEnablement:                genericoptions.NewAPIEnablementOptions(),
				Metrics:                      &basemetrics.Options{},
				ServiceAccountSigningKeyFile: "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.options.Validate()
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}
