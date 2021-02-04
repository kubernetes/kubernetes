/*
Copyright 2021 The Kubernetes Authors.

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

package lint

import (
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/lint"
	"k8s.io/kubectl/pkg/util/i18n"
)

//LintOptions options for lint subcommand
type LintOptions struct {
	FileNameOpts resource.FilenameOptions

	RecordFlags *genericclioptions.RecordFlags
	Recorder    genericclioptions.Recorder

	DynamicClient dynamic.Interface
	Mapper        meta.RESTMapper
	Result        *resource.Result

	genericclioptions.IOStreams
}

//NewLintOptions constructor to type LintOptions
func NewLintOptions(streams genericclioptions.IOStreams) *LintOptions {
	return &LintOptions{
		FileNameOpts: resource.FilenameOptions{},
		RecordFlags:  genericclioptions.NewRecordFlags(),
		Recorder:     genericclioptions.NoopRecorder{},
		IOStreams:    streams,
	}
}

//NewCmdLint builds cobra command
func NewCmdLint(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewLintOptions(streams)
	c := &lint.Cmd{Factory: f, FileNameOptions: &o.FileNameOpts, IOStreams: streams}
	cmd := &cobra.Command{
		Use:                   "lint [-f FILENAME] [-k DIRECTORY]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Lint resource configuration files."),
		Long: i18n.T(`
Look for common issues with resource configuration.  Emit an error message if kubernetes best practices not followed and exit non-0.
`),
		Example: i18n.T(`# lint example.yaml and exit non-0 if issues found.
kubectl lint -f example.yaml
`),
		RunE: func(cmd *cobra.Command, args []string) error {
			return c.Run()
		},
	}
	o.RecordFlags.AddFlags(cmd)
	cmdutil.AddFilenameOptionFlags(cmd, &o.FileNameOpts, "full path to Kubernetes manifests (.yaml) to lint")
	return cmd
}
