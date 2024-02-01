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

package options

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/kubernetes/pkg/kubelet/config"
)

func newKubeletServerOrDie() *KubeletServer {
	s, err := NewKubeletServer()
	if err != nil {
		panic(err)
	}
	return s
}

// TestRoundTrip ensures that flag values from the Kubelet can be serialized
// to arguments and then read back and have the same value. Also catches cases
// where the default value reported by the flag is not actually allowed to be
// specified.
func TestRoundTrip(t *testing.T) {
	testCases := []struct {
		name          string
		inputFlags    func() *KubeletServer
		outputFlags   func() *KubeletServer
		flagDefaulter func(*pflag.FlagSet)
		skipDefault   bool
		err           bool
		expectArgs    bool
	}{
		{
			name:          "default flags are eliminated",
			inputFlags:    newKubeletServerOrDie,
			outputFlags:   newKubeletServerOrDie,
			flagDefaulter: newKubeletServerOrDie().AddFlags,
			err:           false,
			expectArgs:    false,
		},
		{
			name:          "default flag values round trip",
			inputFlags:    newKubeletServerOrDie,
			outputFlags:   newKubeletServerOrDie,
			flagDefaulter: func(*pflag.FlagSet) {},
			err:           false,
			expectArgs:    true,
		},
		{
			name: "nil address does not fail for optional argument",
			inputFlags: func() *KubeletServer {
				s := newKubeletServerOrDie()
				s.HealthzBindAddress = ""
				return s
			},
			outputFlags: func() *KubeletServer {
				s := newKubeletServerOrDie()
				s.HealthzBindAddress = ""
				return s
			},
			flagDefaulter: func(*pflag.FlagSet) {},
			err:           false,
			expectArgs:    true,
		},
	}
	for _, testCase := range testCases {
		modifiedFlags := testCase.inputFlags()
		args := asArgs(modifiedFlags.AddFlags, testCase.flagDefaulter)
		if testCase.expectArgs != (len(args) > 0) {
			t.Errorf("%s: unexpected args: %v", testCase.name, args)
			continue
		}
		t.Logf("%s: args: %v", testCase.name, args)
		flagSet := pflag.NewFlagSet("output", pflag.ContinueOnError)
		outputFlags := testCase.outputFlags()
		outputFlags.AddFlags(flagSet)
		if err := flagSet.Parse(args); err != nil {
			if !testCase.err {
				t.Errorf("%s: unexpected flag error: %v", testCase.name, err)
			}
			continue
		}
		if !reflect.DeepEqual(modifiedFlags, outputFlags) {
			t.Errorf("%s: flags did not round trip: %s", testCase.name, cmp.Diff(modifiedFlags, outputFlags))
			continue
		}
	}
}

func asArgs(fn, defaultFn func(*pflag.FlagSet)) []string {
	fs := pflag.NewFlagSet("extended", pflag.ContinueOnError)
	fn(fs)
	defaults := pflag.NewFlagSet("defaults", pflag.ContinueOnError)
	defaultFn(defaults)
	var args []string
	fs.VisitAll(func(flag *pflag.Flag) {
		// if the flag implements cliflag.OmitEmpty and the value is Empty, then just omit it from the command line
		if omit, ok := flag.Value.(cliflag.OmitEmpty); ok && omit.Empty() {
			return
		}
		s := flag.Value.String()
		// if the flag has the same value as the default, we can omit it without changing the meaning of the command line
		var defaultValue string
		if defaultFlag := defaults.Lookup(flag.Name); defaultFlag != nil {
			defaultValue = defaultFlag.Value.String()
			if s == defaultValue {
				return
			}
		}
		// if the flag is a string slice, each element is specified with an independent flag invocation
		if values, err := fs.GetStringSlice(flag.Name); err == nil {
			for _, s := range values {
				args = append(args, fmt.Sprintf("--%s=%s", flag.Name, s))
			}
		} else {
			if len(s) == 0 {
				s = defaultValue
			}
			args = append(args, fmt.Sprintf("--%s=%s", flag.Name, s))
		}
	})
	return args
}

func TestValidateKubeletFlags(t *testing.T) {
	tests := []struct {
		error       bool
		errorString string
		name        string
		labels      map[string]string
		annotaitons map[string]string
	}{
		{
			name:        "Invalid kubernetes.io label",
			errorString: "unknown 'kubernetes.io' or 'k8s.io' labels specified with --node-labels: [beta.kubernetes.io/metadata-proxy-ready]\n--node-labels in the 'kubernetes.io' namespace must begin with an allowed prefix (kubelet.kubernetes.io, node.kubernetes.io) or be in the specifically allowed set (beta.kubernetes.io/arch, beta.kubernetes.io/instance-type, beta.kubernetes.io/os, failure-domain.beta.kubernetes.io/region, failure-domain.beta.kubernetes.io/zone, kubernetes.io/arch, kubernetes.io/hostname, kubernetes.io/os, node.kubernetes.io/instance-type, topology.kubernetes.io/region, topology.kubernetes.io/zone)",
			error:       true,
			annotaitons: map[string]string{},
			labels: map[string]string{
				"beta.kubernetes.io/metadata-proxy-ready": "true",
			},
		},
		{
			name:        "Valid label outside of kubernetes.io and k8s.io",
			error:       false,
			annotaitons: map[string]string{},
			labels: map[string]string{
				"cloud.google.com/metadata-proxy-ready": "true",
			},
		},
		{
			name:        "Empty label and annotation list",
			error:       false,
			annotaitons: map[string]string{},
			labels:      map[string]string{},
		},
		{
			name:        "Invalid label",
			error:       true,
			errorString: "invalid node labels: 'kubernetes/kubernetes' - a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')",
			annotaitons: map[string]string{},
			labels: map[string]string{
				"cloud.google.com/repository": "kubernetes/kubernetes",
			},
		},
		{
			name:        "Invalid annotaiton",
			error:       true,
			errorString: "invalid node annotaion: 'invalid annotation' - name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
			annotaitons: map[string]string{
				"invalid annotation": "",
			},
			labels: map[string]string{},
		},
		{
			name:        "Multiple annotaitons",
			error:       true,
			errorString: "expected only one annotation set on the node object, got 2",
			annotaitons: map[string]string{
				"annotation1": "",
				"annotation2": "",
			},
			labels: map[string]string{},
		},
		{
			name:        "Disallowed annotaiton",
			errorString: "unexpected 'kubernetes.io' or 'k8s.io' annotations specified with --node-annotations: [beta.kubernetes.io/metadata-proxy-ready]\n--node-annotations in the core components namespace are not allowed",
			error:       true,
			annotaitons: map[string]string{
				"beta.kubernetes.io/metadata-proxy-ready": "true",
			},
			labels: map[string]string{},
		},
		{
			name:  "Valid annotaiton outside of kubernetes.io and k8s.io",
			error: false,
			annotaitons: map[string]string{
				"some.storage.io/default-node-tags": "[\"storage\",\"select\"]",
			},
			labels: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateKubeletFlags(&KubeletFlags{
				ContainerRuntimeOptions: config.ContainerRuntimeOptions{},
				NodeLabels:              tt.labels,
				NodeAnnotation:          tt.annotaitons,
			})

			if tt.error && err == nil {
				t.Errorf("ValidateKubeletFlags should have failed with values: labels=%+v annotations=%+v, causing error: %s, got: %#v", tt.labels, tt.annotaitons, tt.errorString, err)
			}
			if tt.error && err != nil && err.Error() != tt.errorString {
				t.Errorf("ValidateKubeletFlags should have failed with values: labels=%+v annotations=%+v, causing error: %s, got: %#v", tt.labels, tt.annotaitons, tt.errorString, err)
			}

			if !tt.error && err != nil {
				t.Errorf("ValidateKubeletFlags should not have failed with values: labels=%+v annotations=%+v, failed with: %#v", tt.labels, tt.annotaitons, err)
			}
		})
	}

}
