/*
Copyright 2014 Google Inc. All rights reserved.

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
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/spf13/cobra"
)

const (
	updatePeriod       = "1m0s"
	timeout            = "5m0s"
	pollInterval       = "3s"
	rollingUpdate_long = `Perform a rolling update of the given ReplicationController.

Replaces the specified controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.`
	rollingUpdate_example = `// Update pods of frontend-v1 using new controller data in frontend-v2.json.
$ kubectl rolling-update frontend-v1 -f frontend-v2.json

// Update pods of frontend-v1 using JSON data passed into stdin.
$ cat frontend-v2.json | kubectl rolling-update frontend-v1 -f -`
)

func NewCmdRollingUpdate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "rolling-update OLD_CONTROLLER_NAME -f NEW_CONTROLLER_SPEC",
		// rollingupdate is deprecated.
		Aliases: []string{"rollingupdate"},
		Short:   "Perform a rolling update of the given ReplicationController.",
		Long:    rollingUpdate_long,
		Example: rollingUpdate_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunRollingUpdate(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().String("update-period", updatePeriod, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("poll-interval", pollInterval, `Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("timeout", timeout, `Max time to wait for a controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to create the new controller.")
	return cmd
}

func RunRollingUpdate(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if os.Args[1] == "rollingupdate" {
		printDeprecationWarning("rolling-update", "rollingupdate")
	}

	filename := cmdutil.GetFlagString(cmd, "filename")
	if len(filename) == 0 {
		return cmdutil.UsageError(cmd, "Must specify filename for new controller")
	}
	period := cmdutil.GetFlagDuration(cmd, "update-period")
	interval := cmdutil.GetFlagDuration(cmd, "poll-interval")
	timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	if len(args) != 1 {
		return cmdutil.UsageError(cmd, "Must specify the controller to update")
	}
	oldName := args[0]

	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	// TODO: use resource.Builder instead
	obj, err := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		NamespaceParam(cmdNamespace).RequireNamespace().
		FilenameParam(filename).
		Do().
		Object()
	if err != nil {
		return err
	}
	newRc, ok := obj.(*api.ReplicationController)
	if !ok {
		return cmdutil.UsageError(cmd, "%s does not specify a valid ReplicationController", filename)
	}
	newName := newRc.Name
	if oldName == newName {
		return cmdutil.UsageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
			filename, oldName)
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	updater := kubectl.NewRollingUpdater(newRc.Namespace, kubectl.NewRollingUpdaterClient(client))

	// fetch rc
	oldRc, err := client.ReplicationControllers(newRc.Namespace).Get(oldName)
	if err != nil {
		return err
	}

	var hasLabel bool
	for key, oldValue := range oldRc.Spec.Selector {
		if newValue, ok := newRc.Spec.Selector[key]; ok && newValue != oldValue {
			hasLabel = true
			break
		}
	}
	if !hasLabel {
		return cmdutil.UsageError(cmd, "%s must specify a matching key with non-equal value in Selector for %s",
			filename, oldName)
	}
	// TODO: handle resizes during rolling update
	if newRc.Spec.Replicas == 0 {
		newRc.Spec.Replicas = oldRc.Spec.Replicas
	}
	err = updater.Update(&kubectl.RollingUpdaterConfig{
		Out:           out,
		OldRc:         oldRc,
		NewRc:         newRc,
		UpdatePeriod:  period,
		Interval:      interval,
		Timeout:       timeout,
		CleanupPolicy: kubectl.DeleteRollingUpdateCleanupPolicy,
	})
	if err != nil {
		return err
	}

	fmt.Fprintf(out, "%s\n", newName)
	return nil
}
