// Copyright 2018 Microsoft Corporation
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

package cmd

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/spf13/cobra"
)

var depCmd = &cobra.Command{
	Use:   "dep",
	Short: "Calls dep command to execute dep ensure -update",
	Long:  "This command will invoke the dep ensure -update command",
	RunE: func(cmd *cobra.Command, args []string) error {
		err := theDepCommand()
		return err
	},
}

func init() {
	rootCmd.AddCommand(depCmd)
}

func theDepCommand() error {
	println("Executing dep ensure...")
	depArgs := "ensure -update"
	if verboseFlag {
		depArgs += " -v"
	}
	c := exec.Command("dep", strings.Split(depArgs, " ")...)
	err := startCmd(c)
	if err != nil {
		return fmt.Errorf("failed to start command: %v", err)
	}
	return c.Wait()
}
