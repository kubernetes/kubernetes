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

package cmd

import (
	"bytes"
	"reflect"
	"testing"

	psp "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreatePSP(t *testing.T) {
	pspName := "my-psp"

	tf := cmdtesting.NewTestFactory()
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfigVal = defaultClientConfig()

	tests := map[string]struct {
		seLinux            string
		runAsUser          string
		supplementalGroups string
		fsGroup            string
		expectedOutput     *psp.PodSecurityPolicy
	}{
		"empty param": {
			seLinux:            "",
			runAsUser:          "",
			supplementalGroups: "",
			fsGroup:            "",
			expectedOutput: &psp.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: psp.PodSecurityPolicySpec{
					SELinux: psp.SELinuxStrategyOptions{
						Rule: psp.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: psp.RunAsUserStrategyOptions{
						Rule: psp.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: psp.SupplementalGroupsStrategyOptions{
						Rule: psp.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: psp.FSGroupStrategyOptions{
						Rule: psp.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"selinux": {
			seLinux:            "RunAsAny",
			runAsUser:          "",
			supplementalGroups: "",
			fsGroup:            "",
			expectedOutput: &psp.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: psp.PodSecurityPolicySpec{
					SELinux: psp.SELinuxStrategyOptions{
						Rule: psp.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: psp.RunAsUserStrategyOptions{
						Rule: psp.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: psp.SupplementalGroupsStrategyOptions{
						Rule: psp.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: psp.FSGroupStrategyOptions{
						Rule: psp.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"run as user": {
			seLinux:            "",
			runAsUser:          "MustRunAsNonRoot",
			supplementalGroups: "",
			fsGroup:            "",
			expectedOutput: &psp.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: psp.PodSecurityPolicySpec{
					SELinux: psp.SELinuxStrategyOptions{
						Rule: psp.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: psp.RunAsUserStrategyOptions{
						Rule: psp.RunAsUserStrategy("MustRunAsNonRoot"),
					},
					SupplementalGroups: psp.SupplementalGroupsStrategyOptions{
						Rule: psp.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: psp.FSGroupStrategyOptions{
						Rule: psp.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"all": {
			seLinux:            "RunAsAny",
			runAsUser:          "RunAsAny",
			supplementalGroups: "RunAsAny",
			fsGroup:            "RunAsAny",
			expectedOutput: &psp.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: psp.PodSecurityPolicySpec{
					SELinux: psp.SELinuxStrategyOptions{
						Rule: psp.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: psp.RunAsUserStrategyOptions{
						Rule: psp.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: psp.SupplementalGroupsStrategyOptions{
						Rule: psp.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: psp.FSGroupStrategyOptions{
						Rule: psp.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			buf := bytes.NewBuffer([]byte{})
			cmd := NewCmdCreatePSP(tf, buf)
			cmd.Flags().Set("dry-run", "true")
			cmd.Flags().Set("output", "yaml")
			if test.seLinux != "" {
				cmd.Flags().Set("selinux", test.seLinux)
			}
			if test.runAsUser != "" {
				cmd.Flags().Set("run-as-user", test.runAsUser)
			}
			if test.supplementalGroups != "" {
				cmd.Flags().Set("supplemental-groups", test.supplementalGroups)
			}
			if test.fsGroup != "" {
				cmd.Flags().Set("fs-group", test.fsGroup)
			}

			cmd.Run(cmd, []string{pspName})
			actual := &psp.PodSecurityPolicy{}
			if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), buf.Bytes(), actual); err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(test.expectedOutput, actual) {
				t.Errorf("%s:expected %v\n but got %v\n", name, test.expectedOutput, actual)
			}
		})
	}
}
