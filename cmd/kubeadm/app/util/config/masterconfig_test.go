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

package config

import (
	"bytes"
	"io/ioutil"
	"testing"

	"github.com/pmezard/go-difflib/difflib"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const (
	master_v1alpha1YAML                = "testdata/conversion/master/v1alpha1.yaml"
	master_v1alpha1WithoutTypeMetaYAML = "testdata/conversion/master/v1alpha1_without_TypeMeta.yaml"
	master_v1alpha2YAML                = "testdata/conversion/master/v1alpha2.yaml"
	master_internalYAML                = "testdata/conversion/master/internal.yaml"
	master_incompleteYAML              = "testdata/defaulting/master/incomplete.yaml"
	master_defaultedYAML               = "testdata/defaulting/master/defaulted.yaml"
	master_invalidYAML                 = "testdata/validation/invalid_mastercfg.yaml"
	master_beforeUpgradeYAML           = "testdata/v1alpha1_upgrade/before.yaml"
	master_afterUpgradeYAML            = "testdata/v1alpha1_upgrade/after.yaml"
)

func diff(expected, actual []byte) string {
	// Write out the diff
	var diffBytes bytes.Buffer
	difflib.WriteUnifiedDiff(&diffBytes, difflib.UnifiedDiff{
		A:        difflib.SplitLines(string(expected)),
		B:        difflib.SplitLines(string(actual)),
		FromFile: "expected",
		ToFile:   "actual",
		Context:  3,
	})
	return diffBytes.String()
}

func TestConfigFileAndDefaultsToInternalConfig(t *testing.T) {
	var tests = []struct {
		name, in, out string
		groupVersion  schema.GroupVersion
		expectedErr   bool
	}{
		// These tests are reading one file, loading it using ConfigFileAndDefaultsToInternalConfig that all of kubeadm is using for unmarshal of our API types,
		// and then marshals the internal object to the expected groupVersion
		{ // v1alpha1 (faulty) -> internal
			name:         "v1alpha1WithoutTypeMetaToInternal",
			in:           master_v1alpha1WithoutTypeMetaYAML,
			out:          master_internalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		{ // v1alpha1 -> internal
			name:         "v1alpha1ToInternal",
			in:           master_v1alpha1YAML,
			out:          master_internalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		{ // v1alpha2 -> internal
			name:         "v1alpha2ToInternal",
			in:           master_v1alpha2YAML,
			out:          master_internalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		{ // v1alpha1 (faulty) -> internal -> v1alpha2
			name:         "v1alpha1WithoutTypeMetaTov1alpha2",
			in:           master_v1alpha1WithoutTypeMetaYAML,
			out:          master_v1alpha2YAML,
			groupVersion: v1alpha2.SchemeGroupVersion,
		},
		{ // v1alpha1 -> internal -> v1alpha2
			name:         "v1alpha1Tov1alpha2",
			in:           master_v1alpha1YAML,
			out:          master_v1alpha2YAML,
			groupVersion: v1alpha2.SchemeGroupVersion,
		},
		// These tests are reading one file that has only a subset of the fields populated, loading it using ConfigFileAndDefaultsToInternalConfig,
		// and then marshals the internal object to the expected groupVersion
		{ // v1alpha1 (faulty) -> default -> validate -> internal -> v1alpha2
			name:         "incompleteYAMLToDefaultedv1alpha2",
			in:           master_incompleteYAML,
			out:          master_defaultedYAML,
			groupVersion: v1alpha2.SchemeGroupVersion,
		},
		{ // v1alpha1 (faulty) -> validation should fail
			name:        "invalidYAMLShouldFail",
			in:          master_invalidYAML,
			expectedErr: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			internalcfg, err := ConfigFileAndDefaultsToInternalConfig(rt.in, &v1alpha2.MasterConfiguration{})
			if err != nil {
				if rt.expectedErr {
					return
				}
				t2.Fatalf("couldn't unmarshal test data: %v", err)
			}

			actual, err := kubeadmutil.MarshalToYamlForCodecs(internalcfg, rt.groupVersion, scheme.Codecs)
			if err != nil {
				t2.Fatalf("couldn't marshal internal object: %v", err)
			}

			expected, err := ioutil.ReadFile(rt.out)
			if err != nil {
				t2.Fatalf("couldn't read test data: %v", err)
			}

			if !bytes.Equal(expected, actual) {
				t2.Errorf("the expected and actual output differs.\n\tin: %s\n\tout: %s\n\tgroupversion: %s\n\tdiff: \n%s\n",
					rt.in, rt.out, rt.groupVersion.String(), diff(expected, actual))
			}
		})
	}
}

// TestUpgrade tests reading a faulty YAML representation of the MasterConfiguration object (as found in kubeadm clusters <= v1.9.x),
// fixes the problems internally and verifies the marshalled output is the expected output
func TestUpgrade(t *testing.T) {
	before, err := ioutil.ReadFile(master_beforeUpgradeYAML)
	if err != nil {
		t.Fatalf("couldn't read test data: %v", err)
	}

	afterExpected, err := ioutil.ReadFile(master_afterUpgradeYAML)
	if err != nil {
		t.Fatalf("couldn't read test data: %v", err)
	}

	decoded, err := kubeadmutil.LoadYAML(before)
	if err != nil {
		t.Fatalf("couldn't unmarshal test yaml: %v", err)
	}

	scheme := runtime.NewScheme()
	v1alpha1.AddToScheme(scheme)
	codecs := serializer.NewCodecFactory(scheme)

	obj := &v1alpha1.MasterConfiguration{}
	if err := v1alpha1.Migrate(decoded, obj, codecs); err != nil {
		t.Fatalf("couldn't decode migrated object: %v", err)
	}

	afterActual, err := kubeadmutil.MarshalToYamlForCodecs(obj, v1alpha1.SchemeGroupVersion, codecs)
	if err != nil {
		t.Fatalf("couldn't marshal object: %v", err)
	}

	if !bytes.Equal(afterExpected, afterActual) {
		t.Errorf("v1alpha1 object after unmarshal, conversion and marshal didn't match expected value.\n\tdiff: \n%s\n", diff(afterExpected, afterActual))
	}
}

func TestLowercaseSANs(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		out  []string
	}{
		{
			name: "empty struct",
		},
		{
			name: "already lowercase",
			in:   []string{"example.k8s.io"},
			out:  []string{"example.k8s.io"},
		},
		{
			name: "ip addresses and uppercase",
			in:   []string{"EXAMPLE.k8s.io", "10.100.0.1"},
			out:  []string{"example.k8s.io", "10.100.0.1"},
		},
		{
			name: "punycode and uppercase",
			in:   []string{"xn--7gq663byk9a.xn--fiqz9s", "ANOTHEREXAMPLE.k8s.io"},
			out:  []string{"xn--7gq663byk9a.xn--fiqz9s", "anotherexample.k8s.io"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &v1alpha2.MasterConfiguration{
				APIServerCertSANs: test.in,
			}

			LowercaseSANs(cfg.APIServerCertSANs)

			if len(cfg.APIServerCertSANs) != len(test.out) {
				t.Fatalf("expected %d elements, got %d", len(test.out), len(cfg.APIServerCertSANs))
			}

			for i, expected := range test.out {
				if cfg.APIServerCertSANs[i] != expected {
					t.Errorf("expected element %d to be %q, got %q", i, expected, cfg.APIServerCertSANs[i])
				}
			}
		})
	}
}
