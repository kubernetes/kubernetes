// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import "github.com/spf13/cobra"

var (
	// Not using alias because we want 'rkt app exec' appears as
	// a subcommand of 'rkt app'.
	cmdAppExec = &cobra.Command{
		Use:   "exec [--app=APP_NAME] UUID [CMD [ARGS ...]]",
		Short: "Execute commands in the given app's namespace.",
		Long:  "This executes the commands in the given app's namespace. The UUID is the UUID of a running pod. the app name is specified by --app. If CMD and ARGS are empty, then it will execute '/bin/bash' by default.",
		Run:   ensureSuperuser(runWrapper(runEnter)),
	}
)

func init() {
	cmdApp.AddCommand(cmdAppExec)
	cmdAppExec.Flags().StringVar(&flagAppName, "app", "", "name of the app to exec within the specified pod, can be empty if there is only one app in the pod.")
	// Disable interspersed flags to stop parsing after the first non flag
	// argument. This is need to permit to correctly handle
	// ARGS for the CMD.
	cmdAppExec.Flags().SetInterspersed(false)
}
