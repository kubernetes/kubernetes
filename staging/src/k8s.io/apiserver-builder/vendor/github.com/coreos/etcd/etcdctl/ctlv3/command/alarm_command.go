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

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
)

// NewAlarmCommand returns the cobra command for "alarm".
func NewAlarmCommand() *cobra.Command {
	ac := &cobra.Command{
		Use:   "alarm <subcommand>",
		Short: "Alarm related commands",
	}

	ac.AddCommand(NewAlarmDisarmCommand())
	ac.AddCommand(NewAlarmListCommand())

	return ac
}

func NewAlarmDisarmCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "disarm",
		Short: "Disarms all alarms",
		Run:   alarmDisarmCommandFunc,
	}
	return &cmd
}

// alarmDisarmCommandFunc executes the "alarm disarm" command.
func alarmDisarmCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("alarm disarm command accepts no arguments"))
	}
	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).AlarmDisarm(ctx, &v3.AlarmMember{})
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}
	display.Alarm(*resp)
}

func NewAlarmListCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "list",
		Short: "Lists all alarms",
		Run:   alarmListCommandFunc,
	}
	return &cmd
}

// alarmListCommandFunc executes the "alarm list" command.
func alarmListCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("alarm list command accepts no arguments"))
	}
	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).AlarmList(ctx)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}
	display.Alarm(*resp)
}
