// Copyright 2016 Nippon Telegraph and Telephone Corporation.
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

package command

import (
	"fmt"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

// NewAuthCommand returns the cobra command for "auth".
func NewAuthCommand() *cobra.Command {
	ac := &cobra.Command{
		Use:   "auth <enable or disable>",
		Short: "Enable or disable authentication.",
	}

	ac.AddCommand(NewAuthEnableCommand())

	return ac
}

func NewAuthEnableCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "enable",
		Short: "enable authentication",
		Run:   authEnableCommandFunc,
	}
}

// authEnableCommandFunc executes the "auth enable" command.
func authEnableCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("auth enable command does not accept argument."))
	}

	_, err := mustClientFromCmd(cmd).Auth.AuthEnable(context.TODO())
	if err != nil {
		ExitWithError(ExitError, err)
	}
}
