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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
)

const (
	updatePeriod       = "1m0s"
	timeout            = "5m0s"
	pollInterval       = "3s"
	rollingupdate_long = `Perform a rolling update of the given ReplicationController.

Replaces the specified controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.`
	rollingupdate_example = `// Update pods of frontend-v1 using new controller data in frontend-v2.json.
$ kubectl rollingupdate frontend-v1 -f frontend-v2.json

// Update pods of frontend-v1 using JSON data passed into stdin.
$ cat frontend-v2.json | kubectl rollingupdate frontend-v1 -f -`
)

func (f *Factory) NewCmdRollingUpdate(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "rollingupdate OLD_CONTROLLER_NAME -f NEW_CONTROLLER_SPEC",
		Short:   "Perform a rolling update of the given ReplicationController.",
		Long:    rollingupdate_long,
		Example: rollingupdate_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunRollingUpdate(f, out, cmd, args)
			util.CheckErr(err)
		},
	}
	cmd.Flags().String("update-period", updatePeriod, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("poll-interval", pollInterval, `Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("timeout", timeout, `Max time to wait for a controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to create the new controller.")
	return cmd
}

func RunRollingUpdate(f *Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	filename := util.GetFlagString(cmd, "filename")
	if len(filename) == 0 {
		return util.UsageError(cmd, "Must specify filename for new controller")
	}
	period := util.GetFlagDuration(cmd, "update-period")
	interval := util.GetFlagDuration(cmd, "poll-interval")
	timeout := util.GetFlagDuration(cmd, "timeout")
	if len(args) != 1 {
		return util.UsageError(cmd, "Must specify the controller to update")
	}
	oldName := args[0]
	schema, err := f.Validator(cmd)
	if err != nil {
		return err
	}

	clientConfig, err := f.ClientConfig(cmd)
	if err != nil {
		return err
	}
	cmdApiVersion := clientConfig.Version

	mapper, typer := f.Object(cmd)
	// TODO: use resource.Builder instead
	mapping, namespace, newName, data, err := util.ResourceFromFile(filename, typer, mapper, schema, cmdApiVersion)
	if err != nil {
		return err
	}
	if mapping.Kind != "ReplicationController" {
		return util.UsageError(cmd, "%s does not specify a valid ReplicationController", filename)
	}
	if oldName == newName {
		return util.UsageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
			filename, oldName)
	}

	cmdNamespace, err := f.DefaultNamespace(cmd)
	if err != nil {
		return err
	}
	// TODO: use resource.Builder instead
	err = util.CompareNamespace(cmdNamespace, namespace)
	if err != nil {
		return err
	}

	client, err := f.Client(cmd)
	if err != nil {
		return err
	}

	obj, err := mapping.Codec.Decode(data)
	if err != nil {
		return err
	}
	newRc := obj.(*api.ReplicationController)

	updater := kubectl.NewRollingUpdater(cmdNamespace, client)

	// fetch rc
	oldRc, err := client.ReplicationControllers(cmdNamespace).Get(oldName)
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
		return util.UsageError(cmd, "%s must specify a matching key with non-equal value in Selector for %s",
			filename, oldName)
	}
	// TODO: handle resizes during rolling update
	if newRc.Spec.Replicas == 0 {
		newRc.Spec.Replicas = oldRc.Spec.Replicas
	}
	err = updater.Update(out, oldRc, newRc, period, interval, timeout)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, "%s\n", newName)
	return nil
}
