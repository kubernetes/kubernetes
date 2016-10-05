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

	"github.com/renstrom/dedent"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/interrupt"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/spf13/cobra"
)

var (
	status_long = dedent.Dedent(`
		Show the status of the newest rollout.

		By default 'rollout status' will watch the status of the newest rollout
		until it's done. If you don't want to wait for the rollout to finish then
		you can use --watch=false.`)
	status_example = dedent.Dedent(`
		# Watch the rollout status of a deployment
		kubectl rollout status deployment/nginx`)
)

func NewCmdRolloutStatus(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	validArgs := []string{"deployment"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "status (TYPE NAME | TYPE/NAME) [flags]",
		Short:   "Show the status of newest rollout",
		Long:    status_long,
		Example: status_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunStatus(f, cmd, out, args, options))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	cmd.Flags().BoolP("watch", "w", true, "Watch the status of the newest rollout until it's done.")
	return cmd
}

func RunStatus(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string, options *resource.FilenameOptions) error {
	if len(args) == 0 && cmdutil.IsFilenameEmpty(options.Filenames) {
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	mapper, typer := f.Object()

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		ResourceTypeOrNameArgs(true, args...).
		SingleResourceType().
		Latest().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) != 1 {
		return fmt.Errorf("rollout status is only supported on individual resources and resource collections - %d resources were found", len(infos))
	}
	info := infos[0]
	mapping := info.ResourceMapping()

	obj, err := r.Object()
	if err != nil {
		return err
	}
	rv, err := mapping.MetadataAccessor.ResourceVersion(obj)
	if err != nil {
		return err
	}

	statusViewer, err := f.StatusViewer(mapping)
	if err != nil {
		return err
	}

	// check if deployment's has finished the rollout
	status, done, err := statusViewer.Status(cmdNamespace, info.Name)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%s", status)
	if done {
		return nil
	}

	shouldWatch := cmdutil.GetFlagBool(cmd, "watch")
	if !shouldWatch {
		return nil
	}

	// watch for changes to the deployment
	w, err := r.Watch(rv)
	if err != nil {
		return err
	}

	// if the rollout isn't done yet, keep watching deployment status
	intr := interrupt.New(nil, w.Stop)
	return intr.Run(func() error {
		_, err := watch.Until(0, w, func(e watch.Event) (bool, error) {
			// print deployment's status
			status, done, err := statusViewer.Status(cmdNamespace, info.Name)
			if err != nil {
				return false, err
			}
			fmt.Fprintf(out, "%s", status)
			// Quit waiting if the rollout is done
			if done {
				return true, nil
			}
			return false, nil
		})
		return err
	})
}
