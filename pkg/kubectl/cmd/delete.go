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
	"strings"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	delete_long = templates.LongDesc(i18n.T(`
		Delete resources by filenames, stdin, resources and names, or by resources and label selector.

		JSON and YAML formats are accepted. Only one type of the arguments may be specified: filenames,
		resources and names, or resources and label selector.

		Some resources, such as pods, support graceful deletion. These resources define a default period
		before they are forcibly terminated (the grace period) but you may override that value with
		the --grace-period flag, or pass --now to set a grace-period of 1. Because these resources often
		represent entities in the cluster, deletion may not be acknowledged immediately. If the node
		hosting a pod is down or cannot reach the API server, termination may take significantly longer
		than the grace period. To force delete a resource, you must pass a grace period of 0 and specify
		the --force flag.

		IMPORTANT: Force deleting pods does not wait for confirmation that the pod's processes have been
		terminated, which can leave those processes running until the node detects the deletion and
		completes graceful deletion. If your processes use shared storage or talk to a remote API and
		depend on the name of the pod to identify themselves, force deleting those pods may result in
		multiple processes running on different machines using the same identification which may lead
		to data corruption or inconsistency. Only force delete pods when you are sure the pod is
		terminated, or if your application can tolerate multiple copies of the same pod running at once.
		Also, if you force delete pods the scheduler may place new pods on those nodes before the node
		has released those resources and causing those pods to be evicted immediately.

		Note that the delete command does NOT do resource version checks, so if someone submits an
		update to a resource right when you submit a delete, their update will be lost along with the
		rest of the resource.`))

	delete_example = templates.Examples(i18n.T(`
		# Delete a pod using the type and name specified in pod.json.
		kubectl delete -f ./pod.json

		# Delete a pod based on the type and name in the JSON passed into stdin.
		cat pod.json | kubectl delete -f -

		# Delete pods and services with same names "baz" and "foo"
		kubectl delete pod,service baz foo

		# Delete pods and services with label name=myLabel.
		kubectl delete pods,services -l name=myLabel

		# Delete a pod with minimal delay
		kubectl delete pod foo --now

		# Force delete a pod on a dead node
		kubectl delete pod foo --grace-period=0 --force

		# Delete all pods
		kubectl delete pods --all`))
)

type DeleteOptions struct {
	resource.FilenameOptions

	Selector        string
	DeleteAll       bool
	IgnoreNotFound  bool
	Cascade         bool
	DeleteNow       bool
	ForceDeletion   bool
	WaitForDeletion bool

	Reaper func(mapping *meta.RESTMapping) (kubectl.Reaper, error)

	GracePeriod int
	Timeout     time.Duration

	Output string

	Mapper meta.RESTMapper
	Result *resource.Result

	Out    io.Writer
	ErrOut io.Writer
}

func NewCmdDelete(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	deleteFlags := NewDeleteCommandFlags("containing the resource to delete.")
	validArgs := cmdutil.ValidArgList(f)

	cmd := &cobra.Command{
		Use: "delete ([-f FILENAME] | TYPE [(NAME | -l label | --all)])",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Delete resources by filenames, stdin, resources and names, or by resources and label selector"),
		Long:    delete_long,
		Example: delete_example,
		Run: func(cmd *cobra.Command, args []string) {
			options := deleteFlags.ToOptions(out, errOut)
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))

			if err := options.Complete(f, out, errOut, args, cmd); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(cmd); err != nil {
				cmdutil.CheckErr(cmdutil.UsageErrorf(cmd, err.Error()))
			}
			if err := options.RunDelete(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		SuggestFor: []string{"rm"},
		ValidArgs:  validArgs,
		ArgAliases: kubectl.ResourceAliases(validArgs),
	}

	deleteFlags.AddFlags(cmd)

	// flag-specific output flag, as this command does not depend on PrintFlags
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on, not including uninitialized ones.")

	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *DeleteOptions) Complete(f cmdutil.Factory, out, errOut io.Writer, args []string, cmd *cobra.Command) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.Selector = cmdutil.GetFlagString(cmd, "selector")
	o.Reaper = f.Reaper

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	r := f.NewBuilder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		LabelSelectorParam(o.Selector).
		IncludeUninitialized(includeUninitialized).
		SelectAllParam(o.DeleteAll).
		ResourceTypeOrNameArgs(false, args...).RequireObject(false).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	o.Result = r
	o.Mapper = r.Mapper().RESTMapper

	// Set up writer
	o.Out = out
	o.ErrOut = errOut

	return nil
}

func (o *DeleteOptions) Validate(cmd *cobra.Command) error {
	if o.DeleteAll && len(o.Selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}
	if o.DeleteAll {
		f := cmd.Flags().Lookup("ignore-not-found")
		// The flag should never be missing
		if f == nil {
			return fmt.Errorf("missing --ignore-not-found flag")
		}
		// If the user didn't explicitly set the option, default to ignoring NotFound errors when used with --all
		if !f.Changed {
			o.IgnoreNotFound = true
		}
	}
	if o.DeleteNow {
		if o.GracePeriod != -1 {
			return fmt.Errorf("--now and --grace-period cannot be specified together")
		}
		o.GracePeriod = 1
	}
	if o.GracePeriod == 0 {
		if o.ForceDeletion {
			fmt.Fprintf(o.ErrOut, "warning: Immediate deletion does not wait for confirmation that the running resource has been terminated. The resource may continue to run on the cluster indefinitely.\n")
		} else {
			// To preserve backwards compatibility, but prevent accidental data loss, we convert --grace-period=0
			// into --grace-period=1 and wait until the object is successfully deleted. Users may provide --force
			// to bypass this wait.
			o.WaitForDeletion = true
			o.GracePeriod = 1
		}
	} else if o.ForceDeletion {
		fmt.Fprintf(o.ErrOut, "warning: --force is ignored because --grace-period is not 0.\n")
	}
	return nil
}

func (o *DeleteOptions) RunDelete() error {
	// By default use a reaper to delete all related resources.
	if o.Cascade {
		// TODO(juanvallejo): although o.Result can be accessed from the options
		// it is also passed here so that callers of this method outside of the "delete"
		// command do not have to tack it to the "delete" options as well.
		// Find a cleaner way to approach this.
		return o.ReapResult(o.Result, true, false)
	}
	return o.DeleteResult(o.Result)
}

func (o *DeleteOptions) ReapResult(r *resource.Result, isDefaultDelete, quiet bool) error {
	found := 0
	if o.IgnoreNotFound {
		r = r.IgnoreErrors(errors.IsNotFound)
	}
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		found++
		reaper, err := o.Reaper(info.Mapping)
		if err != nil {
			// If there is no reaper for this resources and the user didn't explicitly ask for stop.
			if kubectl.IsNoSuchReaperError(err) && isDefaultDelete {
				// No client side reaper found. Let the server do cascading deletion.
				return o.cascadingDeleteResource(info)
			}
			return cmdutil.AddSourceToErr("reaping", info.Source, err)
		}
		var options *metav1.DeleteOptions
		if o.GracePeriod >= 0 {
			options = metav1.NewDeleteOptions(int64(o.GracePeriod))
		}
		if err := reaper.Stop(info.Namespace, info.Name, o.Timeout, options); err != nil {
			return cmdutil.AddSourceToErr("stopping", info.Source, err)
		}
		if o.WaitForDeletion {
			if err := waitForObjectDeletion(info, o.Timeout); err != nil {
				return cmdutil.AddSourceToErr("stopping", info.Source, err)
			}
		}
		if !quiet {
			o.PrintObj(info)
		}
		return nil
	})
	if err != nil {
		return err
	}
	if found == 0 {
		fmt.Fprintf(o.Out, "No resources found\n")
	}
	return nil
}

func (o *DeleteOptions) DeleteResult(r *resource.Result) error {
	found := 0
	if o.IgnoreNotFound {
		r = r.IgnoreErrors(errors.IsNotFound)
	}
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		found++

		// if we're here, it means that cascade=false (not the default), so we should orphan as requested
		orphan := true
		options := &metav1.DeleteOptions{}
		if o.GracePeriod >= 0 {
			options = metav1.NewDeleteOptions(int64(o.GracePeriod))
		}
		options.OrphanDependents = &orphan
		return o.deleteResource(info, options)
	})
	if err != nil {
		return err
	}
	if found == 0 {
		fmt.Fprintf(o.Out, "No resources found\n")
	}
	return nil
}

func (o *DeleteOptions) cascadingDeleteResource(info *resource.Info) error {
	falseVar := false
	return o.deleteResource(info, &metav1.DeleteOptions{OrphanDependents: &falseVar})
}

func (o *DeleteOptions) deleteResource(info *resource.Info, deleteOptions *metav1.DeleteOptions) error {
	if err := resource.NewHelper(info.Client, info.Mapping).DeleteWithOptions(info.Namespace, info.Name, deleteOptions); err != nil {
		return cmdutil.AddSourceToErr("deleting", info.Source, err)
	}

	o.PrintObj(info)
	return nil
}

// deletion printing is special because we do not have an object to print.
// This mirrors name printer behavior
func (o *DeleteOptions) PrintObj(info *resource.Info) {
	operation := "deleted"
	groupKind := info.Mapping.GroupVersionKind
	kindString := fmt.Sprintf("%s.%s", strings.ToLower(groupKind.Kind), groupKind.Group)
	if len(groupKind.Group) == 0 {
		kindString = strings.ToLower(groupKind.Kind)
	}

	if o.GracePeriod == 0 {
		operation = "force deleted"
	}

	if o.Output == "name" {
		// -o name: prints resource/name
		fmt.Fprintf(o.Out, "%s/%s\n", kindString, info.Name)
		return
	}

	// understandable output by default
	fmt.Fprintf(o.Out, "%s \"%s\" %s\n", kindString, info.Name, operation)
}

// objectDeletionWaitInterval is the interval to wait between checks for deletion.
var objectDeletionWaitInterval = time.Second

// waitForObjectDeletion refreshes the object, waiting until it is deleted, a timeout is reached, or
// an error is encountered. It checks once a second.
func waitForObjectDeletion(info *resource.Info, timeout time.Duration) error {
	copied := *info
	info = &copied
	// TODO: refactor Reaper so that we can pass the "wait" option into it, and then check for UID change.
	return wait.PollImmediate(objectDeletionWaitInterval, timeout, func() (bool, error) {
		switch err := info.Get(); {
		case err == nil:
			return false, nil
		case errors.IsNotFound(err):
			return true, nil
		default:
			return false, err
		}
	})
}
