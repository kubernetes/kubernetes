/*
Copyright 2024 The Kubernetes Authors.

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

package kuberc

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/component-base/cli/flag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/spf13/cobra"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1alpha1"
)

type fakeCmds struct {
	name  string
	flags []fakeFlag
}

type fakeFlag struct {
	name  string
	value string
}

func TestApplyOverride(t *testing.T) {
	tests := []struct {
		name               string
		nestedCmds         []fakeCmds
		args               []string
		getPreferencesFunc func(kuberc string) (*v1alpha1.Preferences, error)
		expectedFLags      []fakeFlag
	}{
		{
			name: "command override",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "changed",
				},
			},
		},
		{
			name: "subcommand override",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "changed",
				},
			},
		},
		{
			name: "use explicit kuberc, subcommand explicit takes precedence",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--kuberc",
				"test-custom-kuberc-path",
				"--firstflag=explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				if kuberc != "test-custom-kuberc-path" {
					return nil, fmt.Errorf("unexpected kuberc: %s", kuberc)
				}
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
		{
			name: "use explicit kuberc equal, subcommand explicit takes precedence",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--kuberc=test-custom-kuberc-path",
				"--firstflag=explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				if kuberc != "test-custom-kuberc-path" {
					return nil, fmt.Errorf("unexpected kuberc: %s", kuberc)
				}
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
		{
			name: "use explicit kuberc equal at the end, subcommand explicit takes precedence",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--firstflag=explicit",
				"--kuberc=test-custom-kuberc-path",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				if kuberc != "test-custom-kuberc-path" {
					return nil, fmt.Errorf("unexpected kuberc: %s", kuberc)
				}
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
		{
			name: "subcommand explicit takes precedence",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--firstflag=explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
		{
			name: "subcommand explicit takes precedence with space",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--firstflag",
				"explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
		{
			name: "subcommand check the not overridden flag",
			nestedCmds: []fakeCmds{
				{
					name:  "command1",
					flags: nil,
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
						{
							name:  "secondflag",
							value: "secondflagvalue",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--firstflag",
				"explicit",
				"--secondflag=changed",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
				{
					name:  "secondflag",
					value: "changed",
				},
			},
		},
		{
			name: "command1 also has same flag",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "shouldnuse",
						},
						{
							name:  "secondflag",
							value: "shouldnuse",
						},
					},
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"command1",
				"command2",
				"--firstflag",
				"explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Overrides: []v1alpha1.PreferencesCommandOverride{
							{
								Command: "command1 command2",
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cmdtesting.WithAlphaEnvs([]util.FeatureGate{util.KubeRC}, t, func(t *testing.T) {
				rootCmd := &cobra.Command{
					Use: "root",
				}
				prefHandler := NewPreferences()
				prefHandler.AddFlags(rootCmd.PersistentFlags())
				pref, ok := prefHandler.(*Preferences)
				if !ok {
					t.Fatal("unexpected type. Expected *Preferences")
				}
				addCommands(rootCmd, test.nestedCmds)
				pref.GetPreferencesFunc = test.getPreferencesFunc
				errWriter := &bytes.Buffer{}
				err := pref.ApplyOverrides(rootCmd, test.args, errWriter)
				if err != nil {
					t.Fatalf("unexpected error %v\n", err)
				}
				actualCmd, _, err := rootCmd.Find(test.args[1:])
				if err != nil {
					t.Fatalf("unable to find the command %v\n", err)
				}

				err = actualCmd.ParseFlags(test.args[1:])
				if err != nil {
					t.Fatalf("unexpected error %v\n", err)
				}

				if errWriter.String() != "" {
					t.Fatalf("unexpected error message %s\n", errWriter.String())
				}

				for _, expectedFlag := range test.expectedFLags {
					actualFlag := actualCmd.Flag(expectedFlag.name)
					if actualFlag.Value.String() != expectedFlag.value {
						t.Fatalf("unexpected flag value expected %s actual %s", expectedFlag.value, actualFlag.Value.String())
					}
				}
			})
		})
	}
}

func TestApplyAlias(t *testing.T) {
	tests := []struct {
		name               string
		nestedCmds         []fakeCmds
		args               []string
		getPreferencesFunc func(kuberc string) (*v1alpha1.Preferences, error)
		expectedFLags      []fakeFlag
		expectedCmd        string
		expectedArgs       []string
	}{
		{
			name: "command override",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"getcmd",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Aliases: []v1alpha1.PreferencesAliasOverride{
							{
								Name:    "getcmd",
								Command: "command1",
								Arguments: []string{
									"resources",
									"nodes",
								},
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "changed",
				},
			},
			expectedCmd: "getcmd",
			expectedArgs: []string{
				"resources",
				"nodes",
			},
		},
		{
			name: "command override",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"getcmd",
				"--firstflag=explicit",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Aliases: []v1alpha1.PreferencesAliasOverride{
							{
								Name:    "getcmd",
								Command: "command1",
								Arguments: []string{
									"resources",
									"nodes",
								},
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
			},
			expectedCmd: "getcmd",
			expectedArgs: []string{
				"resources",
				"nodes",
			},
		},
		{
			name: "command override",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
						{
							name:  "secondflag",
							value: "secondflagvalue",
						},
					},
				},
			},
			args: []string{
				"root",
				"getcmd",
				"--firstflag=explicit",
				"--secondflag=changed",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Aliases: []v1alpha1.PreferencesAliasOverride{
							{
								Name:    "getcmd",
								Command: "command1",
								Arguments: []string{
									"resources",
									"nodes",
								},
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "explicit",
				},
				{
					name:  "secondflag",
					value: "changed",
				},
			},
			expectedCmd: "getcmd",
			expectedArgs: []string{
				"resources",
				"nodes",
			},
		},
		{
			name: "simple aliasing",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"aliascmd",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Aliases: []v1alpha1.PreferencesAliasOverride{
							{
								Name:    "aliascmd",
								Command: "command1",
								Arguments: []string{
									"resources",
									"nodes",
								},
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "changed",
				},
			},
			expectedCmd: "aliascmd",
			expectedArgs: []string{
				"resources",
				"nodes",
			},
		},
		{
			name: "subcommand aliasing",
			nestedCmds: []fakeCmds{
				{
					name: "command1",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "shouldntuse",
						},
						{
							name:  "secondflag",
							value: "shouldntuse",
						},
					},
				},
				{
					name: "command2",
					flags: []fakeFlag{
						{
							name:  "firstflag",
							value: "test",
						},
					},
				},
			},
			args: []string{
				"root",
				"aliascmd",
			},
			getPreferencesFunc: func(kuberc string) (*v1alpha1.Preferences, error) {
				return &v1alpha1.Preferences{
					TypeMeta: metav1.TypeMeta{},
					Spec: v1alpha1.PreferencesSpec{
						Aliases: []v1alpha1.PreferencesAliasOverride{
							{
								Name:    "aliascmd",
								Command: "command1 command2",
								Arguments: []string{
									"resources",
									"nodes",
								},
								Flags: []v1alpha1.PreferencesCommandOverrideFlag{
									{
										Name:    "firstflag",
										Default: "changed2",
									},
								},
							},
						},
					},
				}, nil
			},
			expectedFLags: []fakeFlag{
				{
					name:  "firstflag",
					value: "changed2",
				},
			},
			expectedCmd: "aliascmd",
			expectedArgs: []string{
				"resources",
				"nodes",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cmdtesting.WithAlphaEnvs([]util.FeatureGate{util.KubeRC}, t, func(t *testing.T) {
				rootCmd := &cobra.Command{
					Use: "root",
				}
				prefHandler := NewPreferences()
				prefHandler.AddFlags(rootCmd.PersistentFlags())
				pref, ok := prefHandler.(*Preferences)
				if !ok {
					t.Fatal("unexpected type. Expected *Preferences")
				}
				addCommands(rootCmd, test.nestedCmds)
				pref.GetPreferencesFunc = test.getPreferencesFunc
				errWriter := &bytes.Buffer{}
				lastArgs, err := pref.ApplyAliases(rootCmd, test.args, errWriter)
				if err != nil {
					t.Fatalf("unexpected error %v\n", err)
				}
				actualCmd, _, err := rootCmd.Find(lastArgs[1:])
				if err != nil {
					t.Fatal(err)
				}

				err = actualCmd.ParseFlags(lastArgs)
				if err != nil {
					t.Fatalf("unexpected error %v\n", err)
				}

				if errWriter.String() != "" {
					t.Fatalf("unexpected error message %s\n", errWriter.String())
				}

				if test.expectedCmd != actualCmd.Name() {
					t.Fatalf("unexpected command expected %s actual %s", test.expectedCmd, actualCmd.Name())
				}

				for _, expectedFlag := range test.expectedFLags {
					actualFlag := actualCmd.Flag(expectedFlag.name)
					if actualFlag.Value.String() != expectedFlag.value {
						t.Fatalf("unexpected flag value expected %s actual %s", expectedFlag.value, actualFlag.Value.String())
					}
				}

				for _, expectedArg := range test.expectedArgs {
					found := false
					for _, actualArg := range lastArgs {
						if actualArg == expectedArg {
							found = true
							break
						}
					}
					if !found {
						t.Fatalf("expected arg %s can not be found", expectedArg)
					}
				}
			})
		})
	}
}

// Add list of commands in nested way.
// First iteration adds command into rootCmd,
// Second iteration adds command into the previous one.
func addCommands(rootCmd *cobra.Command, commands []fakeCmds) {
	if len(commands) == 0 {
		return
	}

	subCmd := &cobra.Command{
		Use: commands[0].name,
	}

	for _, flg := range commands[0].flags {
		val := flag.StringFlag{}
		val.Set(flg.value) // nolint: errcheck
		subCmd.Flags().AddFlag(&pflag.Flag{Name: flg.name, Value: &val})
	}
	rootCmd.AddCommand(subCmd)

	addCommands(subCmd, commands[1:])
}
