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

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// ResumeOptions holds command line options required to run the command.
type ResumeOptions struct {
	resource.FilenameOptions

	Resumer func(object *resource.Info) ([]byte, error)
	Mapper  meta.RESTMapper
	Typer   runtime.ObjectTyper
	Encoder runtime.Encoder
	Infos   []*resource.Info

	Out io.Writer
}

var (
	resumeLong = templates.LongDesc(`
		Resume a paused resource

		Paused resources will not be reconciled by a controller. By resuming a
		resource, we allow it to be reconciled again.
		Currently only deployments support being resumed.`)

	resumeExample = templates.Examples(`
		# Resume an already paused deployment
		kubectl rollout resume deployment/nginx`)
)

// NewCmdRolloutResume creates the `resume` subcommand.
func NewCmdRolloutResume(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ResumeOptions{}

	validArgs := []string{"deployment"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "resume RESOURCE",
		Short:   i18n.T("Resume a paused resource"),
		Long:    resumeLong,
		Example: resumeExample,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := options.Complete(f, cmd, out, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = options.Run()
			if err != nil {
				allErrs = append(allErrs, err)
			}
			cmdutil.CheckErr(utilerrors.Flatten(utilerrors.NewAggregate(allErrs)))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	return cmd
}

// Complete completes all the required options.
func (o *ResumeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		return cmdutil.UsageErrorf(cmd, "%s", cmd.Use)
	}

	o.Mapper, o.Typer = f.Object()
	o.Encoder = f.JSONEncoder()

	o.Resumer = f.Resumer
	o.Out = out

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		o.Infos = append(o.Infos, info)
		return nil
	})
	if err != nil {
		return err
	}
	return nil
}

// Run implements the actual command.
func (o ResumeOptions) Run() error {
	allErrs := []error{}
	for _, patch := range set.CalculatePatches(o.Infos, o.Encoder, o.Resumer) {
		info := patch.Info

		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s %q %v", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "already resumed")
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch: %v", err))
			continue
		}

		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "resumed")
	}

	return utilerrors.NewAggregate(allErrs)
}
