/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/errors"

	"github.com/spf13/cobra"
)

// HistoryOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type HistoryOptions struct {
	Filenames []string
	Recursive bool
}

const (
	history_long    = `View previous rollout revisions and configurations.`
	history_example = `# View the rollout history of a deployment
kubectl rollout history deployment/abc

# View the details of deployment revision 3
kubectl rollout history deployment/abc --revision=3`
)

func NewCmdRolloutHistory(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &HistoryOptions{}

	cmd := &cobra.Command{
		Use:     "history (TYPE NAME | TYPE/NAME) [flags]",
		Short:   "view rollout history",
		Long:    history_long,
		Example: history_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunHistory(f, cmd, out, args, options))
		},
	}

	cmd.Flags().Int64("revision", 0, "See the details, including podTemplate of the revision specified")
	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	return cmd
}

func RunHistory(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string, options *HistoryOptions) error {
	if len(args) == 0 && len(options.Filenames) == 0 {
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}
	revisionDetail := cmdutil.GetFlagInt64(cmd, "revision")

	mapper, typer := f.Object(false)

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	infos, err := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		ResourceTypeOrNameArgs(true, args...).
		Latest().
		Flatten().
		Do().
		Infos()
	if err != nil {
		return err
	}

	errs := []error{}
	for _, info := range infos {
		mapping := info.ResourceMapping()
		historyViewer, err := f.HistoryViewer(mapping)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		historyInfo, err := historyViewer.History(info.Namespace, info.Name)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		if revisionDetail > 0 {
			// Print details of a specific revision
			template, ok := historyInfo.RevisionToTemplate[revisionDetail]
			if !ok {
				return fmt.Errorf("unable to find revision %d of %s %q", revisionDetail, mapping.Resource, info.Name)
			}
			fmt.Fprintf(out, "%s %q revision %d\n", mapping.Resource, info.Name, revisionDetail)
			kubectl.DescribePodTemplate(template, out)
		} else {
			// Print all revisions
			formattedOutput, printErr := kubectl.PrintRolloutHistory(historyInfo, mapping.Resource, info.Name)
			if printErr != nil {
				errs = append(errs, printErr)
				continue
			}
			fmt.Fprintf(out, "%s\n", formattedOutput)
		}
	}

	return errors.NewAggregate(errs)
}
