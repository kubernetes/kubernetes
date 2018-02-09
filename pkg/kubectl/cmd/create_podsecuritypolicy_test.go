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
	"io"
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	pspv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testPodSecurityPolicyPrinter struct {
	CachedPSP *pspv1beta1.PodSecurityPolicy
}

func (t *testPodSecurityPolicyPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedPSP = obj.(*pspv1beta1.PodSecurityPolicy)
	return nil
}

func (t *testPodSecurityPolicyPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testPodSecurityPolicyPrinter) HandledResources() []string {
	return []string{}
}

func (t *testPodSecurityPolicyPrinter) IsGeneric() bool {
	return true
}

func TestCreatePSP(t *testing.T) {
	pspName := "my-cluster-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testPodSecurityPolicyPrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := map[string]struct {
		option         commandOption
		expectedOutput *pspv1beta1.PodSecurityPolicy
	}{
		"basic": {
			option: commandOption{
				seLinux:            "RunAsAny",
				runAsUser:          "RunAsAny",
				supplementalGroups: "RunAsAny",
				fsGroup:            "RunAsAny",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"selinux": {
			option: commandOption{
				seLinux:            "RunAsAny,user=user1,role=role1,type=type1,level=level1",
				runAsUser:          "RunAsAny",
				supplementalGroups: "RunAsAny",
				fsGroup:            "RunAsAny",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
						SELinuxOptions: &v1.SELinuxOptions{
							User:  "user1",
							Role:  "role1",
							Type:  "type1",
							Level: "level1",
						},
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"run-as-user": {
			option: commandOption{
				seLinux:            "RunAsAny",
				runAsUser:          "RunAsAny,100-111,200-222,300-333",
				supplementalGroups: "RunAsAny",
				fsGroup:            "RunAsAny",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
						Ranges: []pspv1beta1.IDRange{
							{
								Min: 100,
								Max: 111,
							},
							{
								Min: 200,
								Max: 222,
							},
							{
								Min: 300,
								Max: 333,
							},
						},
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"supplemental-groups": {
			option: commandOption{
				seLinux:            "RunAsAny",
				runAsUser:          "RunAsAny",
				supplementalGroups: "MustRunAs,0-100,2-200",
				fsGroup:            "RunAsAny",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("MustRunAs"),
						Ranges: []pspv1beta1.IDRange{
							{
								Min: 0,
								Max: 100,
							},
							{
								Min: 2,
								Max: 200,
							},
						},
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
				},
			},
		},

		"fs-groups": {
			option: commandOption{
				seLinux:            "RunAsAny",
				runAsUser:          "RunAsAny",
				supplementalGroups: "RunAsAny",
				fsGroup:            "MustRunAs,20000-30000",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("MustRunAs"),
						Ranges: []pspv1beta1.IDRange{
							{
								Min: 20000,
								Max: 30000,
							},
						},
					},
				},
			},
		},

		"host-ports": {
			option: commandOption{
				seLinux:            "RunAsAny",
				runAsUser:          "MustRunAsNonRoot",
				supplementalGroups: "RunAsAny",
				fsGroup:            "RunAsAny",
				hostPorts:          "1234-4321,5000-6000",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("MustRunAsNonRoot"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
					HostPorts: []pspv1beta1.HostPortRange{
						{
							Min: 1234,
							Max: 4321,
						},
						{
							Min: 5000,
							Max: 6000,
						},
					},
				},
			},
		},

		"other": {
			option: commandOption{
				privileged:                      "true",
				defaultAddCapabilities:          "/dev/default1,/dev/default2",
				requiredDropCapabilities:        "/dev/drop1,/dev/drop2",
				allowedCapabilities:             "/dev/allow1,/dev/allow2",
				volumes:                         "emptyDir",
				hostNetwork:                     "true",
				hostPorts:                       "1-1000",
				hostPID:                         "true",
				hostIPC:                         "true",
				seLinux:                         "RunAsAny",
				runAsUser:                       "RunAsAny",
				supplementalGroups:              "RunAsAny",
				fsGroup:                         "RunAsAny",
				readOnlyRootFilesystem:          "true",
				defaultAllowPrivilegeEscalation: "true",
				allowPrivilegeEscalation:        "true",
				allowedHostPaths:                "/dev/allowedpath1,/dev/allowedpath2,/dev/allowedpath3",
				allowedFlexVolumes:              "/dev/allowedflexvolume1,/dev/allowedflexvolume2,/dev/allowedflexvolume3",
			},
			expectedOutput: &pspv1beta1.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: pspName,
				},
				Spec: pspv1beta1.PodSecurityPolicySpec{
					Privileged: true,
					DefaultAddCapabilities: []v1.Capability{
						v1.Capability("/dev/default1"),
						v1.Capability("/dev/default2"),
					},
					RequiredDropCapabilities: []v1.Capability{
						v1.Capability("/dev/drop1"),
						v1.Capability("/dev/drop2"),
					},
					AllowedCapabilities: []v1.Capability{
						v1.Capability("/dev/allow1"),
						v1.Capability("/dev/allow2"),
					},
					Volumes: []pspv1beta1.FSType{
						pspv1beta1.FSType("emptyDir"),
					},
					HostNetwork: true,
					HostPorts: []pspv1beta1.HostPortRange{
						{
							Min: 1,
							Max: 1000,
						},
					},
					HostPID: true,
					HostIPC: true,
					SELinux: pspv1beta1.SELinuxStrategyOptions{
						Rule: pspv1beta1.SELinuxStrategy("RunAsAny"),
					},
					RunAsUser: pspv1beta1.RunAsUserStrategyOptions{
						Rule: pspv1beta1.RunAsUserStrategy("RunAsAny"),
					},
					SupplementalGroups: pspv1beta1.SupplementalGroupsStrategyOptions{
						Rule: pspv1beta1.SupplementalGroupsStrategyType("RunAsAny"),
					},
					FSGroup: pspv1beta1.FSGroupStrategyOptions{
						Rule: pspv1beta1.FSGroupStrategyType("RunAsAny"),
					},
					ReadOnlyRootFilesystem:          true,
					DefaultAllowPrivilegeEscalation: getBoolP(true),
					AllowPrivilegeEscalation:        getBoolP(true),
					AllowedHostPaths: []pspv1beta1.AllowedHostPath{
						{
							PathPrefix: "/dev/allowedpath1",
						},
						{
							PathPrefix: "/dev/allowedpath2",
						},
						{
							PathPrefix: "/dev/allowedpath3",
						},
					},
					AllowedFlexVolumes: []pspv1beta1.AllowedFlexVolume{
						{
							Driver: "/dev/allowedflexvolume1",
						},
						{
							Driver: "/dev/allowedflexvolume2",
						},
						{
							Driver: "/dev/allowedflexvolume3",
						},
					},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			buf := bytes.NewBuffer([]byte{})
			cmd := NewCmdCreatePSP(f, buf)
			cmd.Flags().Set("dry-run", "true")
			cmd.Flags().Set("output", "object")

			setCommand(cmd, test.option)
			cmd.Run(cmd, []string{pspName})
			if !reflect.DeepEqual(test.expectedOutput, printer.CachedPSP) {
				t.Errorf("%s:%s", name, diff.ObjectReflectDiff(test.expectedOutput, printer.CachedPSP))
			}
		})
	}
}

func TestNewStrategyOptions(t *testing.T) {
	tests := map[string]struct {
		rules          []string
		expectedOutput strategyOptions
		expectedError  bool
	}{
		"single-rule": {
			rules: []string{"RunAsAny"},
			expectedOutput: strategyOptions{
				strategyRule: "RunAsAny",
			},
			expectedError: false,
		},
		"double-rule": {
			rules: []string{"RunAsAny", "100-1000"},
			expectedOutput: strategyOptions{
				strategyRule: "RunAsAny",
				ranges: []pspv1beta1.IDRange{
					{Min: 100, Max: 1000},
				},
			},
			expectedError: false,
		},
		"trible-rule": {
			rules: []string{"RunAsAny", "100-1000", "4000-4001"},
			expectedOutput: strategyOptions{
				strategyRule: "RunAsAny",
				ranges: []pspv1beta1.IDRange{
					{Min: 100, Max: 1000},
					{Min: 4000, Max: 4001},
				},
			},
			expectedError: false,
		},
		"error port format-1": {
			rules:          []string{"RunAsAny", "100:1000", "4000-4001"},
			expectedOutput: strategyOptions{},
			expectedError:  true,
		},
		"error port format-2": {
			rules:          []string{"RunAsAny", "test1-test2", "4000-4001"},
			expectedOutput: strategyOptions{},
			expectedError:  true,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			stra, err := newStrategyOptions("run-as-user", test.rules)
			if err != nil {
				if !test.expectedError {
					t.Errorf("unexpected error:%s", err)
				}
				return
			}

			if test.expectedError {
				t.Errorf("%s: expect error happens but passes.", name)
			}

			if !reflect.DeepEqual(test.expectedOutput, stra) {
				t.Errorf("%s:%s", name, diff.ObjectReflectDiff(test.expectedOutput, stra))
			}
		})
	}
}

func TestNewSELinuxOptions(t *testing.T) {
	tests := map[string]struct {
		rules          []string
		expectedOutput pspv1beta1.SELinuxStrategyOptions
	}{
		"single-rule": {
			rules: []string{"RunAsAny"},
			expectedOutput: pspv1beta1.SELinuxStrategyOptions{
				Rule: "RunAsAny",
			},
		},
		"double-rule": {
			rules: []string{"MustRunAs", "role=r"},
			expectedOutput: pspv1beta1.SELinuxStrategyOptions{
				Rule: "MustRunAs",
				SELinuxOptions: &v1.SELinuxOptions{
					Role: "r",
				},
			},
		},
		"trible-rule": {
			rules: []string{"RunAsAny", "user=u", "role=r", "type=t", "level=l"},
			expectedOutput: pspv1beta1.SELinuxStrategyOptions{
				Rule: "RunAsAny",
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "u",
					Role:  "r",
					Type:  "t",
					Level: "l",
				},
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			se, err := newSELinuxOptions(test.rules)
			if err != nil {
				t.Errorf("test %s faild: %v", err)
			}
			if !reflect.DeepEqual(test.expectedOutput, se) {
				t.Errorf("%s:%s", name, diff.ObjectReflectDiff(test.expectedOutput, se))
			}
		})
	}
}

type commandOption struct {
	privileged                      string
	defaultAddCapabilities          string
	requiredDropCapabilities        string
	allowedCapabilities             string
	volumes                         string
	hostNetwork                     string
	hostPorts                       string
	hostPID                         string
	hostIPC                         string
	seLinux                         string
	runAsUser                       string
	supplementalGroups              string
	fsGroup                         string
	readOnlyRootFilesystem          string
	defaultAllowPrivilegeEscalation string
	allowPrivilegeEscalation        string
	allowedHostPaths                string
	allowedFlexVolumes              string
}

func getBoolP(b bool) *bool {
	var ret bool
	ret = b
	return &ret
}

func setCommand(cmd *cobra.Command, option commandOption) {
	if option.privileged != "" {
		cmd.Flags().Set("privileged", option.privileged)
	}
	if option.defaultAddCapabilities != "" {
		cmd.Flags().Set("default-add-cap", option.defaultAddCapabilities)
	}
	if option.requiredDropCapabilities != "" {
		cmd.Flags().Set("required-drop-cap", option.requiredDropCapabilities)
	}
	if option.allowedCapabilities != "" {
		cmd.Flags().Set("allowed-cap", option.allowedCapabilities)
	}
	if option.volumes != "" {
		cmd.Flags().Set("volumes", option.volumes)
	}
	if option.hostNetwork != "" {
		cmd.Flags().Set("host-network", option.hostNetwork)
	}
	if option.hostPorts != "" {
		cmd.Flags().Set("host-ports", option.hostPorts)
	}
	if option.hostPID != "" {
		cmd.Flags().Set("host-pid", option.hostPID)
	}
	if option.hostIPC != "" {
		cmd.Flags().Set("host-ipc", option.hostIPC)
	}
	if option.seLinux != "" {
		cmd.Flags().Set("selinux", option.seLinux)
	}
	if option.runAsUser != "" {
		cmd.Flags().Set("run-as-user", option.runAsUser)
	}
	if option.supplementalGroups != "" {
		cmd.Flags().Set("supplemental-groups", option.supplementalGroups)
	}
	if option.fsGroup != "" {
		cmd.Flags().Set("fs-group", option.fsGroup)
	}
	if option.readOnlyRootFilesystem != "" {
		cmd.Flags().Set("readonly-root-fs", option.readOnlyRootFilesystem)
	}
	if option.defaultAllowPrivilegeEscalation != "" {
		cmd.Flags().Set("default-allow-privilege-escalation", option.defaultAllowPrivilegeEscalation)
	}
	if option.allowPrivilegeEscalation != "" {
		cmd.Flags().Set("allow-privilege-escalation", option.allowPrivilegeEscalation)
	}
	if option.allowedHostPaths != "" {
		cmd.Flags().Set("allowed-host-paths", option.allowedHostPaths)
	}
	if option.allowedFlexVolumes != "" {
		cmd.Flags().Set("allowed-flex-volumes", option.allowedFlexVolumes)
	}
}
