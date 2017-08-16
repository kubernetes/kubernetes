/*
Copyright 2016 The Kubernetes Authors.

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

package rollout

import (
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"github.com/spf13/cobra"
)

var (
	history_long = templates.LongDesc(`
		View previous rollout revisions and configurations.`)

	history_example = templates.Examples(`
		# View the rollout history of a deployment
		kubectl rollout history deployment/abc

		# View the details of daemonset revision 3
		kubectl rollout history daemonset/abc --revision=3`)
)

func NewCmdRolloutHistory(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	validArgs := []string{"deployment", "daemonset"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "history (TYPE NAME | TYPE/NAME) [flags]",
		Short:   i18n.T("View rollout history"),
		Long:    history_long,
		Example: history_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunHistory(f, cmd, out, args, options))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	cmd.Flags().Int64("revision", 0, "See the details, including podTemplate of the revision specified")
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	return cmd
}

func RunHistory(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string, options *resource.FilenameOptions) error {
	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(options.Filenames) {
		return cmdutil.UsageErrorf(cmd, "Required resource not specified.")
	}
	revision := cmdutil.GetFlagInt64(cmd, "revision")
	if revision < 0 {
		return fmt.Errorf("revision must be a positive integer: %v", revision)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder(true).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		mapping := info.ResourceMapping()
		historyViewer, err := f.HistoryViewer(mapping)
		if err != nil {
			return err
		}
		historyInfo, err := historyViewer.ViewHistory(info.Namespace, info.Name, revision)
		if err != nil {
			return err
		}

		header := fmt.Sprintf("%s %q", mapping.Resource, info.Name)
		if revision > 0 {
			header = fmt.Sprintf("%s with revision #%d", header, revision)
		}
		fmt.Fprintf(out, "%s\n", header)
		fmt.Fprintf(out, "%s\n", historyInfo)
		return nil
	})
}
