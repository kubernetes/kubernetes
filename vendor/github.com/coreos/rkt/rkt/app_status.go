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

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	rkt "github.com/coreos/rkt/lib"
	"github.com/spf13/cobra"
)

var (
	cmdAppStatus = &cobra.Command{
		Use:   "status UUID --app=APP_NAME [--format=json]",
		Short: "Check the status of an app in the given pod",
		Long:  "This will print detailed status of an app",
		Run:   runWrapper(runAppStatus),
	}
)

func init() {
	cmdApp.AddCommand(cmdAppStatus)
	cmdAppStatus.Flags().StringVar(&flagAppName, "app", "", "choose app within the pod, this flag must be set")
	cmdAppStatus.Flags().Var(&flagFormat, "format", "choose the output format, allowed format includes 'json', 'json-pretty'. If empty, then the result is printed as key value pairs")
}

func printApp(app *rkt.App) {
	stdout.Printf("name=%s\n", app.Name)
	stdout.Printf("state=%s\n", app.State)
	stdout.Printf("image_id=%s\n", app.ImageID)
	if app.CreatedAt != nil {
		stdout.Printf("created_at=%v\n", time.Unix(0, *(app.CreatedAt)))
	}
	if app.StartedAt != nil {
		stdout.Printf("started_at=%v\n", time.Unix(0, *(app.StartedAt)))
	}
	if app.FinishedAt != nil {
		stdout.Printf("finished_at=%v\n", time.Unix(0, *(app.FinishedAt)))
	}
	if app.ExitCode != nil {
		stdout.Printf("exit_code=%d\n", *(app.ExitCode))
	}

	if len(app.Mounts) > 0 {
		stdout.Printf("mounts=")
		var mnts []string
		for _, mnt := range app.Mounts {
			mnts = append(mnts, fmt.Sprintf("%s:%s:(read_only:%v)", mnt.HostPath, mnt.ContainerPath, mnt.ReadOnly))
		}
		stdout.Printf(strings.Join(mnts, ","))
		stdout.Println()
	}

	if len(app.UserAnnotations) > 0 {
		stdout.Printf("user_annotations=")
		var annos []string
		for key, value := range app.UserAnnotations {
			annos = append(annos, fmt.Sprintf("%s:%s", key, value))
		}
		stdout.Printf(strings.Join(annos, ","))
		stdout.Println()
	}

	if len(app.UserLabels) > 0 {
		stdout.Printf("user_labels=")
		var labels []string
		for key, value := range app.UserLabels {
			labels = append(labels, fmt.Sprintf("%s:%s", key, value))
		}
		stdout.Printf(strings.Join(labels, ","))
		stdout.Println()
	}
}

func runAppStatus(cmd *cobra.Command, args []string) (exit int) {
	if len(args) != 1 || flagAppName == "" {
		cmd.Usage()
		return 1
	}

	apps, err := rkt.AppsForPod(args[0], getDataDir(), flagAppName)
	if err != nil {
		stderr.PrintE("error getting app status", err)
		return 1
	}

	if len(apps) == 0 {
		stderr.Error(fmt.Errorf("cannot find app %q in the pod", flagAppName))
		return 1
	}

	// Must have only 1 app.
	if len(apps) != 1 {
		stderr.Error(fmt.Errorf("find more than one app with the name %q", flagAppName))
		return 1
	}

	// TODO(yifan): Print yamls.
	switch flagFormat {
	case outputFormatJSON:
		result, err := json.Marshal(apps[0])
		if err != nil {
			stderr.PrintE("error marshaling the app status", err)
			return 1
		}
		stdout.Print(string(result))
	case outputFormatPrettyJSON:
		result, err := json.MarshalIndent(apps[0], "", "\t")
		if err != nil {
			stderr.PrintE("error marshaling the app status", err)
			return 1
		}
		stdout.Print(string(result))
	default:
		printApp(apps[0])
	}

	return 0
}
