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
	"bytes"
	"fmt"

	rkt "github.com/coreos/rkt/lib"
	"github.com/spf13/cobra"
)

var (
	cmdAppList = &cobra.Command{
		Use:   "list UUID",
		Short: "List apps for the given pod",
		Long:  "This only lists the name and state of the apps, app status will show more detailed info.",
		Run:   runWrapper(runAppList),
	}
)

func init() {
	cmdApp.AddCommand(cmdAppList)
	cmdAppList.Flags().BoolVar(&flagNoLegend, "no-legend", false, "suppress a legend with the list")
}

func runAppList(cmd *cobra.Command, args []string) int {
	if len(args) != 1 {
		cmd.Usage()
		return 1
	}

	apps, err := rkt.AppsForPod(args[0], getDataDir(), "")
	if err != nil {
		stderr.PrintE("error listing apps", err)
		return 1
	}

	tabBuffer := new(bytes.Buffer)
	tabOut := getTabOutWithWriter(tabBuffer)

	if !flagNoLegend {
		fmt.Fprintf(tabOut, "NAME\tSTATE\n")
	}

	for _, app := range apps {
		fmt.Fprintf(tabOut, "%s\t%s\n", app.Name, app.State)
	}

	tabOut.Flush()
	stdout.Print(tabBuffer)
	return 0
}
