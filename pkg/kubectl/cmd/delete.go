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
	"fmt"
	"io"
	"time"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// DeleteOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type DeleteOptions struct {
	Filenames []string
	Recursive bool
}

var (
	delete_long = dedent.Dedent(`
		Delete resources by filenames, stdin, resources and names, or by resources and label selector.

		JSON and YAML formats are accepted.

		Only one type of the arguments may be specified: filenames, resources and names, or resources and label selector

		Note that the delete command does NOT do resource version checks, so if someone
		submits an update to a resource right when you submit a delete, their update
		will be lost along with the rest of the resource.`)
	delete_example = dedent.Dedent(`
		# Delete a pod using the type and name specified in pod.json.
		kubectl delete -f ./pod.json

		# Delete a pod based on the type and name in the JSON passed into stdin.
		cat pod.json | kubectl delete -f -

		# Delete pods and services with same names "baz" and "foo"
		kubectl delete pod,service baz foo

		# Delete pods and services with label name=myLabel.
		kubectl delete pods,services -l name=myLabel

		# Delete a pod immediately (no graceful shutdown)
		kubectl delete pod foo --now

		# Delete a pod with UID 1234-56-7890-234234-456456.
		kubectl delete pod 1234-56-7890-234234-456456

		# Delete all pods
		kubectl delete pods --all`)
)

func NewCmdDelete(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &DeleteOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "delete ([-f FILENAME] | TYPE [(NAME | -l label | --all)])",
		Short:   "Delete resources by filenames, stdin, resources and names, or by resources and label selector",
		Long:    delete_long,
		Example: delete_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			err := RunDelete(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		SuggestFor: []string{"rm"},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	usage := "Filename, directory, or URL to a file containing the resource to delete."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on.")
	cmd.Flags().Bool("all", false, "[-all] to select all the specified resources.")
	cmd.Flags().Bool("ignore-not-found", false, "Treat \"resource not found\" as a successful delete. Defaults to \"true\" when --all is specified.")
	cmd.Flags().Bool("cascade", true, "If true, cascade the deletion of the resources managed by this resource (e.g. Pods created by a ReplicationController).  Default true.")
	cmd.Flags().Int("grace-period", -1, "Period of time in seconds given to the resource to terminate gracefully. Ignored if negative.")
	cmd.Flags().Bool("now", false, "If true, resources are force terminated without graceful deletion (same as --grace-period=0).")
	cmd.Flags().Duration("timeout", 0, "The length of time to wait before giving up on a delete, zero means determine a timeout from the size of the object")
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func RunDelete(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *DeleteOptions) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	deleteAll := cmdutil.GetFlagBool(cmd, "all")
	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		SelectorParam(cmdutil.GetFlagString(cmd, "selector")).
		SelectAllParam(deleteAll).
		ResourceTypeOrNameArgs(false, args...).RequireObject(false).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	ignoreNotFound := cmdutil.GetFlagBool(cmd, "ignore-not-found")
	if deleteAll {
		f := cmd.Flags().Lookup("ignore-not-found")
		// The flag should never be missing
		if f == nil {
			return fmt.Errorf("missing --ignore-not-found flag")
		}
		// If the user didn't explicitly set the option, default to ignoring NotFound errors when used with --all
		if !f.Changed {
			ignoreNotFound = true
		}
	}

	gracePeriod := cmdutil.GetFlagInt(cmd, "grace-period")
	if cmdutil.GetFlagBool(cmd, "now") {
		if gracePeriod != -1 {
			return fmt.Errorf("--now and --grace-period cannot be specified together")
		}
		gracePeriod = 0
	}

	shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
	// By default use a reaper to delete all related resources.
	if cmdutil.GetFlagBool(cmd, "cascade") {
		return ReapResult(r, f, out, cmdutil.GetFlagBool(cmd, "cascade"), ignoreNotFound, cmdutil.GetFlagDuration(cmd, "timeout"), gracePeriod, shortOutput, mapper, false)
	}
	return DeleteResult(r, out, ignoreNotFound, shortOutput, mapper)
}

func ReapResult(r *resource.Result, f *cmdutil.Factory, out io.Writer, isDefaultDelete, ignoreNotFound bool, timeout time.Duration, gracePeriod int, shortOutput bool, mapper meta.RESTMapper, quiet bool) error {
	found := 0
	if ignoreNotFound {
		r = r.IgnoreErrors(errors.IsNotFound)
	}
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		found++
		reaper, err := f.Reaper(info.Mapping)
		if err != nil {
			// If there is no reaper for this resources and the user didn't explicitly ask for stop.
			if kubectl.IsNoSuchReaperError(err) && isDefaultDelete {
				return deleteResource(info, out, shortOutput, mapper)
			}
			return cmdutil.AddSourceToErr("reaping", info.Source, err)
		}
		var options *api.DeleteOptions
		if gracePeriod >= 0 {
			options = api.NewDeleteOptions(int64(gracePeriod))
		}
		if err := reaper.Stop(info.Namespace, info.Name, timeout, options); err != nil {
			return cmdutil.AddSourceToErr("stopping", info.Source, err)
		}
		if !quiet {
			cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, false, "deleted")
		}
		return nil
	})
	if err != nil {
		return err
	}
	if found == 0 {
		fmt.Fprintf(out, "No resources found\n")
	}
	return nil
}

func DeleteResult(r *resource.Result, out io.Writer, ignoreNotFound bool, shortOutput bool, mapper meta.RESTMapper) error {
	found := 0
	if ignoreNotFound {
		r = r.IgnoreErrors(errors.IsNotFound)
	}
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		found++
		return deleteResource(info, out, shortOutput, mapper)
	})
	if err != nil {
		return err
	}
	if found == 0 {
		fmt.Fprintf(out, "No resources found\n")
	}
	return nil
}

func deleteResource(info *resource.Info, out io.Writer, shortOutput bool, mapper meta.RESTMapper) error {
	if err := resource.NewHelper(info.Client, info.Mapping).Delete(info.Namespace, info.Name); err != nil {
		return cmdutil.AddSourceToErr("deleting", info.Source, err)
	}
	cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, false, "deleted")
	return nil
}
