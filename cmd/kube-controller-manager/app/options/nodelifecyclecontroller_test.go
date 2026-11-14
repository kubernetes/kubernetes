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

package options

import (
	"strings"
	"testing"

	"github.com/spf13/pflag"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	"k8s.io/kubernetes/pkg/features"
)

func TestNodeLifecycleControllerOptions_AddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	opts := &NodeLifecycleControllerOptions{
		NodeLifecycleControllerConfiguration: &nodelifecycleconfig.NodeLifecycleControllerConfiguration{
			NodeLivenessSource: nodelifecycleconfig.NodeLivenessSourceKubelet,
		},
	}

	opts.AddFlags(fs)

	flag := fs.Lookup("node-liveness-source")
	if flag == nil {
		t.Fatal("node-liveness-source flag was not added")
	}
	if flag.DefValue != nodelifecycleconfig.NodeLivenessSourceKubelet {
		t.Errorf("expected default value %q, got %q", nodelifecycleconfig.NodeLivenessSourceKubelet, flag.DefValue)
	}

	if err := fs.Parse([]string{"--node-liveness-source=external"}); err != nil {
		t.Fatalf("failed to parse flags: %v", err)
	}
	if opts.NodeLivenessSource != nodelifecycleconfig.NodeLivenessSourceExternal {
		t.Errorf("expected NodeLivenessSource %q, got %q", nodelifecycleconfig.NodeLivenessSourceExternal, opts.NodeLivenessSource)
	}
}

func TestNodeLifecycleControllerOptions_Validate(t *testing.T) {
	testCases := []struct {
		name                   string
		gateEnabled            bool
		nodeLivenessSource     string
		expectErrors           bool
		expectedErrorSubString string
	}{
		{
			name:               "gate off, empty defaults to kubelet",
			gateEnabled:        false,
			nodeLivenessSource: "",
			expectErrors:       false,
		},
		{
			name:               "gate off, explicit kubelet",
			gateEnabled:        false,
			nodeLivenessSource: nodelifecycleconfig.NodeLivenessSourceKubelet,
			expectErrors:       false,
		},
		{
			name:                   "gate off, external rejected",
			gateEnabled:            false,
			nodeLivenessSource:     nodelifecycleconfig.NodeLivenessSourceExternal,
			expectErrors:           true,
			expectedErrorSubString: "ExternalNodeLivenessDetection",
		},
		{
			name:                   "gate off, unknown value rejected",
			gateEnabled:            false,
			nodeLivenessSource:     "custom",
			expectErrors:           true,
			expectedErrorSubString: `must be "kubelet" or "external"`,
		},
		{
			name:               "gate on, empty defaults to kubelet",
			gateEnabled:        true,
			nodeLivenessSource: "",
			expectErrors:       false,
		},
		{
			name:               "gate on, explicit kubelet",
			gateEnabled:        true,
			nodeLivenessSource: nodelifecycleconfig.NodeLivenessSourceKubelet,
			expectErrors:       false,
		},
		{
			name:               "gate on, external allowed",
			gateEnabled:        true,
			nodeLivenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			expectErrors:       false,
		},
		{
			name:                   "gate on, unknown value rejected",
			gateEnabled:            true,
			nodeLivenessSource:     "custom",
			expectErrors:           true,
			expectedErrorSubString: `must be "kubelet" or "external"`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, tc.gateEnabled)
			opts := &NodeLifecycleControllerOptions{
				NodeLifecycleControllerConfiguration: &nodelifecycleconfig.NodeLifecycleControllerConfiguration{
					NodeLivenessSource: tc.nodeLivenessSource,
				},
			}

			errs := opts.Validate()
			if tc.expectErrors && len(errs) == 0 {
				t.Fatal("expected validation errors, got none")
			}
			if !tc.expectErrors && len(errs) > 0 {
				t.Fatalf("expected no validation errors, got: %v", errs)
			}
			if tc.expectErrors {
				gotErr := utilerrors.NewAggregate(errs).Error()
				if !strings.Contains(gotErr, tc.expectedErrorSubString) {
					t.Errorf("expected error containing %q, got %q", tc.expectedErrorSubString, gotErr)
				}
			}
		})
	}
}

func TestNodeLifecycleControllerOptions_Nil(t *testing.T) {
	var nilOpts *NodeLifecycleControllerOptions
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	nilOpts.AddFlags(fs)
	if flag := fs.Lookup("node-liveness-source"); flag != nil {
		t.Error("expected no flag registered for nil options")
	}
	if errs := nilOpts.Validate(); len(errs) != 0 {
		t.Errorf("expected no errors for nil options, got %v", errs)
	}
	emptyOpts := &NodeLifecycleControllerOptions{}
	if errs := emptyOpts.Validate(); len(errs) != 0 {
		t.Errorf("expected no errors for nil embedded config, got %v", errs)
	}
}
