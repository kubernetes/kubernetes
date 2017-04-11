// Copyright 2016 The etcd Authors
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
)

// NewAuthCommand returns the cobra command for "auth".
func NewAuthCommand() *cobra.Command {
	ac := &cobra.Command{
		Use:   "auth <enable or disable>",
		Short: "Enable or disable authentication",
	}

	ac.AddCommand(newAuthEnableCommand())
	ac.AddCommand(newAuthDisableCommand())

	return ac
}

func newAuthEnableCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "enable",
		Short: "Enables authentication",
		Run:   authEnableCommandFunc,
	}
}

// authEnableCommandFunc executes the "auth enable" command.
func authEnableCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("auth enable command does not accept any arguments."))
	}

	ctx, cancel := commandCtx(cmd)
	_, err := mustClientFromCmd(cmd).Auth.AuthEnable(ctx)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Println("Authentication Enabled")
}

func newAuthDisableCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "disable",
		Short: "Disables authentication",
		Run:   authDisableCommandFunc,
	}
}

// authDisableCommandFunc executes the "auth disable" command.
func authDisableCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("auth disable command does not accept any arguments."))
	}

	ctx, cancel := commandCtx(cmd)
	_, err := mustClientFromCmd(cmd).Auth.AuthDisable(ctx)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Println("Authentication Disabled")
}
