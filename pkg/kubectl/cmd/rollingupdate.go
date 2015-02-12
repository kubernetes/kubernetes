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
	updatePeriod = "1m0s"
	timeout      = "5m0s"
	pollInterval = "3s"
)

func (f *Factory) NewCmdRollingUpdate(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "rollingupdate <old-controller-name> -f <new-controller.json>",
		Short: "Perform a rolling update of the given ReplicationController.",
		Long: `Perform a rolling update of the given ReplicationController.

Replaces the specified controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.

Examples:

    // Update pods of frontend-v1 using new controller data in frontend-v2.json.
    $ kubectl rollingupdate frontend-v1 -f frontend-v2.json

    // Update pods of frontend-v1 using JSON data passed into stdin.
    $ cat frontend-v2.json | kubectl rollingupdate frontend-v1 -f -`,
		Run: func(cmd *cobra.Command, args []string) {
			filename := util.GetFlagString(cmd, "filename")
			if len(filename) == 0 {
				usageError(cmd, "Must specify filename for new controller")
			}
			period := util.GetFlagDuration(cmd, "update-period")
			interval := util.GetFlagDuration(cmd, "poll-interval")
			timeout := util.GetFlagDuration(cmd, "timeout")
			if len(args) != 1 {
				usageError(cmd, "Must specify the controller to update")
			}
			oldName := args[0]
			schema, err := f.Validator(cmd)
			checkErr(err)

			clientConfig, err := f.ClientConfig(cmd)
			checkErr(err)
			cmdApiVersion := clientConfig.Version

			mapper, typer := f.Object(cmd)
			mapping, namespace, newName, data := util.ResourceFromFile(filename, typer, mapper, schema, cmdApiVersion)
			if mapping.Kind != "ReplicationController" {
				usageError(cmd, "%s does not specify a valid ReplicationController", filename)
			}
			if oldName == newName {
				usageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
					filename, oldName)
			}

			cmdNamespace, err := f.DefaultNamespace(cmd)
			checkErr(err)
			err = util.CompareNamespace(cmdNamespace, namespace)
			checkErr(err)

			client, err := f.Client(cmd)
			checkErr(err)

			obj, err := mapping.Codec.Decode(data)
			checkErr(err)
			newRc := obj.(*api.ReplicationController)

			updater := kubectl.NewRollingUpdater(namespace, client)

			// fetch rc
			oldRc, err := client.ReplicationControllers(namespace).Get(oldName)
			checkErr(err)

			var hasLabel bool
			for key, oldValue := range oldRc.Spec.Selector {
				if newValue, ok := newRc.Spec.Selector[key]; ok && newValue != oldValue {
					hasLabel = true
					break
				}
			}
			if !hasLabel {
				usageError(cmd, "%s must specify a matching key with non-equal value in Selector for %s",
					filename, oldName)
			}
			// TODO: handle resizes during rolling update
			if newRc.Spec.Replicas == 0 {
				newRc.Spec.Replicas = oldRc.Spec.Replicas
			}
			err = updater.Update(out, oldRc, newRc, period, interval, timeout)
			checkErr(err)

			fmt.Fprintf(out, "%s\n", newName)
		},
	}
	cmd.Flags().String("update-period", updatePeriod, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("poll-interval", pollInterval, `Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("timeout", timeout, `Max time to wait for a controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to create the new controller.")
	return cmd
}
