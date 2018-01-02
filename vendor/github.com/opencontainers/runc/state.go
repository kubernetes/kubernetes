// +build linux

package main

import (
	"encoding/json"
	"os"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/urfave/cli"
)

var stateCommand = cli.Command{
	Name:  "state",
	Usage: "output the state of a container",
	ArgsUsage: `<container-id>

Where "<container-id>" is your name for the instance of the container.`,
	Description: `The state command outputs current state information for the
instance of a container.`,
	Action: func(context *cli.Context) error {
		if err := checkArgs(context, 1, exactArgs); err != nil {
			return err
		}
		container, err := getContainer(context)
		if err != nil {
			return err
		}
		containerStatus, err := container.Status()
		if err != nil {
			return err
		}
		state, err := container.State()
		if err != nil {
			return err
		}
		pid := state.BaseState.InitProcessPid
		if containerStatus == libcontainer.Stopped {
			pid = 0
		}
		bundle, annotations := utils.Annotations(state.Config.Labels)
		cs := containerState{
			Version:        state.BaseState.Config.Version,
			ID:             state.BaseState.ID,
			InitProcessPid: pid,
			Status:         containerStatus.String(),
			Bundle:         bundle,
			Rootfs:         state.BaseState.Config.Rootfs,
			Created:        state.BaseState.Created,
			Annotations:    annotations,
		}
		data, err := json.MarshalIndent(cs, "", "  ")
		if err != nil {
			return err
		}
		os.Stdout.Write(data)
		return nil
	},
}
