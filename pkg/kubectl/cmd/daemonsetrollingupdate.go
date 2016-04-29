/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"reflect"
	"time"

	"github.com/golang/glog"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// RollingUpdateOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
//type RollingUpdateOptions struct {
//	Filenames []string
//}

const (
	daemonSetRollingUpdate_long = `Perform a rolling update of the given DaemonSet.

Replaces the specified daemon set with a new daemon set by creating a daemon set with the same selector and delering the
existing pods. The new-daemonset.json must specify the same namespace as the
existing daemon set`
	daemonSetRollingUpdate_example = `# Update the daemon set ds-v1 into ds-v1 and recreate associated pod.
$ kubectl daemonset-rolling-update ds-v1 -f ds-v2.json

# Update daemon set ds-v1 using JSON data passed into stdin.
$ cat ds-v2.json | kubectl daemonset-rolling-update ds-v1 -f -

# Abort and reverse an existing rollout in progress (from ds-v1 to ds-v2).
$ kubectl daemonset-rolling-update ds-v1 ds-v2 --rollback
`
)

var (
	recreateInterval, _ = time.ParseDuration("0s")
	deleteInterval, _   = time.ParseDuration("10s")
)

func NewCmdDaemonSetRollingUpdate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &RollingUpdateOptions{}

	cmd := &cobra.Command{
		Use:     "daemonset-rolling-update OLD_DS_NAME -f NEW_DS_SPEC",
		Short:   "Perform a rolling update of the given DaemonSet.",
		Long:    daemonSetRollingUpdate_long,
		Example: daemonSetRollingUpdate_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunDaemonSetRollingUpdate(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().Duration("recreate-interval", recreateInterval,
		`Time to wait between each pod recreation. You may not need that because the update wait for pods to be ready `+
			`before deleting the next one. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().Duration("delete-interval", deleteInterval,
		`Time delay between daemon set creation and deletion of old one. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().Duration("timeout", timeout, `Max time to wait for the full update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	usage := "Filename or URL to file to use to create the new daemon set."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.MarkFlagRequired("filename")
	cmd.Flags().Bool("dry-run", false, "If true, print out the changes that would be made, but don't actually make them.")
	cmd.Flags().Bool("rollback", false, "If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	return cmd
}

func validateDSArguments(cmd *cobra.Command, filenames, args []string) error {

	if len(filenames) > 1 {
		return cmdutil.UsageError(cmd, "May only specify a single filename for new daemon set")
	}

	if len(filenames) == 0 {
		return cmdutil.UsageError(cmd, "Must specify --filename for new daemon set")
	}

	if len(args) < 1 {
		return cmdutil.UsageError(cmd, "Must specify the daemon set to update")
	}

	return nil
}

func RunDaemonSetRollingUpdate(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *RollingUpdateOptions) error {

	var newDs, oldDs *extensions.DaemonSet

	err := validateDSArguments(cmd, options.Filenames, args)
	if err != nil {
		return err
	}

	filename := ""
	oldName := args[0]
	rollback := cmdutil.GetFlagBool(cmd, "rollback")
	rInterval := cmdutil.GetFlagDuration(cmd, "recreate-interval")
	dInterval := cmdutil.GetFlagDuration(cmd, "delete-interval")
	//timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	dryrun := cmdutil.GetFlagBool(cmd, "dry-run")
	outputFormat := cmdutil.GetFlagString(cmd, "output")

	if len(options.Filenames) > 0 {
		filename = options.Filenames[0]
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()

	if len(filename) != 0 {
		schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
		if err != nil {
			return err
		}

		request := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
			Schema(schema).
			NamespaceParam(cmdNamespace).DefaultNamespace().
			FilenameParam(enforceNamespace, filename).
			Do()
		obj, err := request.Object()
		if err != nil {
			return err
		}
		var ok bool
		// Handle filename input from stdin. The resource builder always returns an api.List
		// when creating resource(s) from a stream.
		if list, ok := obj.(*api.List); ok {
			if len(list.Items) > 1 {
				return cmdutil.UsageError(cmd, "%s specifies multiple items", filename)
			}
			obj = list.Items[0]
		}
		newDs, ok = obj.(*extensions.DaemonSet)
		if !ok {
			if gvk, err := typer.ObjectKind(obj); err == nil {
				return cmdutil.UsageError(cmd, "%s contains a %v not a DaemonSet", filename, gvk)
			}
			glog.V(4).Infof("Object %#v is not a DaemonSet", obj)
			return cmdutil.UsageError(cmd, "%s does not specify a valid DaemonSet", filename)
		}
	}

	updater := kubectl.NewDaemonSetRollingUpdater(cmdNamespace, client)

	// fetch ds
	oldDs, err = client.Extensions().DaemonSets(cmdNamespace).Get(oldName)
	if err != nil {
		if !errors.IsNotFound(err) || len(args) > 1 {
			return err
		}

		if rollback {
			fmt.Fprintf(out, "Can't do rollback because the old DS was already deleted in the previous run")
			fmt.Fprintf(out, "If you have your old manifest you can rollback by doing the rolling update the other way")
			return nil
		}
		// Maybe we are continuing an update. Let's check if we have a ds with the
		// oldname as annotation. If so just recreate pods
		newDs, err := client.Extensions().DaemonSets(cmdNamespace).Get(newDs.Name)
		if err != nil {
			return err
		}

		if dryrun {
			fmt.Fprintf(out, "Only recreating pods of %s. Data provided will be ignored because of inconsitent state", newDs.Name)
			return nil
		} else {
			return updater.RecreatePods(newDs, rInterval, out)
		}
	}

	// Same selector is needed
	if !reflect.DeepEqual(*oldDs.Spec.Selector, *newDs.Spec.Selector) {
		return cmdutil.UsageError(cmd, "%s must have the same selector as the existing DaemonSet %s",
			filename, oldName)
	}

	// Different name is needed
	if oldName == newDs.Name {
		return cmdutil.UsageError(cmd, "%s must not have the same name as the existing DaemonSet %s",
			filename, oldName)
	}

	if dryrun {
		oldDsData := &bytes.Buffer{}
		newDsData := &bytes.Buffer{}
		if outputFormat == "" {
			oldDsData.WriteString(oldDs.Name)
			newDsData.WriteString(newDs.Name)
		} else {
			if err := f.PrintObject(cmd, oldDs, oldDsData); err != nil {
				return err
			}
			if err := f.PrintObject(cmd, newDs, newDsData); err != nil {
				return err
			}
		}
		fmt.Fprintf(out, "Rolling from:\n%s\nTo:\n%s\n", string(oldDsData.Bytes()), string(newDsData.Bytes()))
		return nil
	}

	if rollback {

		fmt.Fprintf(out, "Rolling back\n")
		err = updater.DeleteDs(newDs.Name, out)
		if err != nil {
			return err
		}
		return updater.RecreatePods(oldDs, rInterval, out)

	}

	config := &kubectl.DaemonSetRollingUpdaterConfig{
		Out:       out,
		OldDs:     oldDs,
		NewDs:     newDs,
		RInterval: rInterval,
		DInterval: dInterval,
	}

	updater = kubectl.NewDaemonSetRollingUpdater(newDs.Namespace, client)

	err = updater.Update(config)
	if err != nil {
		return err
	}

	message := "rolling updated"
	/*if keepOldName {
		newRc.Name = oldName
	} else {
		message = fmt.Sprintf("rolling updated to %q", newRc.Name)
	}*/
	message = fmt.Sprintf("rolling updated to %q", newDs.Name)
	newDs, err = client.Extensions().DaemonSets(cmdNamespace).Get(newDs.Name)
	if err != nil {
		return err
	}
	/*if outputFormat != "" {
		return f.PrintObject(cmd, newDs, out)
	}*/
	kind, err := api.Scheme.ObjectKind(newDs)
	if err != nil {
		return err
	}
	_, res := meta.KindToResource(kind)
	cmdutil.PrintSuccess(mapper, false, out, res.Resource, oldName, message)
	return nil
}
