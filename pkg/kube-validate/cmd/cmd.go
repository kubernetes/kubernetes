/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/util/yaml"

	"github.com/spf13/cobra"
)

const (
	usage_template = `Usage:
  {{.UseLine}}
`
	help_template = `{{with or .Long .Short }}{{. | trim}}{{end}}`
)

// NewKubeValidateCommand creates the `kube-validate` command.
func NewKubeValidateCommand(in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kube-validate [file] [file] ...",
		Short: "kube-validate validates Kubernetes pod schemas locally",
		Long: `kube-validate validates Kubernetes pod schemas locally.

		kube-validate validates using Version 1 of the Kubernetes API specification.

Find more information at https://github.com/kubernetes/kubernetes.
`,
		Run: runCommand,
	}

	cmds.SetHelpTemplate(help_template)
	cmds.SetUsageTemplate(usage_template)

	return cmds
}

func runCommand(cmd *cobra.Command, args []string) {
	if len(args) == 0 || (len(args) == 1 && (args[0] == "-h" || args[0] == "--help")) {
		cmd.Help()
	} else {
		for _, file := range args {
			if err := validateFile(file); err != nil {
				fmt.Println(err)
			}
		}
	}
}

func validateFile(file string) error {

	var podObj api.Pod
	SetDefaults_Pod(&podObj)

	fileBytes, err := ioutil.ReadFile(file)
	if err != nil {
		return fmt.Errorf("Failed to read file %s: %v", file, err)
	}

	fileBytes, err = yaml.ToJSON(fileBytes)
	if err != nil {
		return fmt.Errorf("Failed to parse YAML/JSON file %s: %v", file, err)
	}

	err = json.Unmarshal(fileBytes, &podObj)
	if err != nil {
		return fmt.Errorf("Failed to convert file %s to valid Pod: %v", file, err)
	}

	if errs := validation.ValidatePod(&podObj); len(errs) != 0 {
		return fmt.Errorf("Invalid Pod: %v", errs)
	}

	return nil
}

// Helper function; not copied from anywhere (its own implementation)
func SetDefaults_Pod(obj *api.Pod) {
	if obj.Namespace == "" {
		obj.Namespace = api.NamespaceDefault
	}
	SetDefaults_PodSpec(&obj.Spec)
}

// Helper function; copied from pkg/api/v1/defaults.go:145
func SetDefaults_PodSpec(obj *api.PodSpec) {
	if obj.DNSPolicy == "" {
		obj.DNSPolicy = api.DNSClusterFirst
	}
	if obj.RestartPolicy == "" {
		obj.RestartPolicy = api.RestartPolicyAlways
	}
	if obj.SecurityContext == nil {
		obj.SecurityContext = &api.PodSecurityContext{}
	}
	if obj.TerminationGracePeriodSeconds == nil {
		period := int64(v1.DefaultTerminationGracePeriodSeconds)
		obj.TerminationGracePeriodSeconds = &period
	}
}
