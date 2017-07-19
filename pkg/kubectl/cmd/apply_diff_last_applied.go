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

package cmd

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/diff"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	applyDiffLastAppliedLong = templates.LongDesc(i18n.T(`
		Opens up a 2-way diff in the default diff viewer. This should follow the same semantics as 'git diff'.
		It should accept an environment variable KUBECTL_EXTERNAL_DIFF=meld to specify a custom diff tool.
		If not specified, the 'git diff' command should be used, if 'git diff' not found, it will degenerate to use diff.
		`))

	applyDiffLastAppliedExample = templates.Examples(i18n.T(`
		# Diff the last-applied-configuration annotations with input file
		kubectl apply diff-last-applied -f deploy.yaml
		`))
)

func NewCmdApplyDiffLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &diff.Options{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "diff-last-applied -f FILENAME",
		Short:   i18n.T("Diff the last-applied-configuration annotation on a live object to match the contents of a file."),
		Long:    applyDiffLastAppliedLong,
		Example: applyDiffLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd))
			cmdutil.CheckErr(options.Validate(f, cmd))
			cmdutil.CheckErr(options.RunDiffLastApplied(f, cmd))
		},
	}
	cmdutil.AddOutputVarFlagsForJsonYaml(cmd, &options.Output, "yaml")
	usage := "that contains the last-applied-configuration annotations"
	kubectl.AddJsonFilenameFlag(cmd, &options.FilenameOptions.Filenames, "Filename, directory, or URL to files "+usage)

	return cmd
}
