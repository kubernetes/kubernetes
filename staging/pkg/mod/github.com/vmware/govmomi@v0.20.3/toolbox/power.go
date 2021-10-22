/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"fmt"
	"log"
	"os/exec"
)

// GuestOsState enum as defined in open-vm-tools/lib/include/vmware/guestrpc/powerops.h
const (
	_ = iota
	powerStateHalt
	powerStateReboot
	powerStatePowerOn
	powerStateResume
	powerStateSuspend
)

var (
	shutdown = "/sbin/shutdown"
)

type PowerCommand struct {
	Handler func() error

	out   *ChannelOut
	state int
	name  string
}

type PowerCommandHandler struct {
	Halt    PowerCommand
	Reboot  PowerCommand
	PowerOn PowerCommand
	Resume  PowerCommand
	Suspend PowerCommand
}

func registerPowerCommandHandler(service *Service) *PowerCommandHandler {
	handler := new(PowerCommandHandler)

	handlers := map[string]struct {
		cmd   *PowerCommand
		state int
	}{
		"OS_Halt":    {&handler.Halt, powerStateHalt},
		"OS_Reboot":  {&handler.Reboot, powerStateReboot},
		"OS_PowerOn": {&handler.PowerOn, powerStatePowerOn},
		"OS_Resume":  {&handler.Resume, powerStateResume},
		"OS_Suspend": {&handler.Suspend, powerStateSuspend},
	}

	for name, h := range handlers {
		*h.cmd = PowerCommand{
			name:  name,
			state: h.state,
			out:   service.out,
		}

		service.RegisterHandler(name, h.cmd.Dispatch)
	}

	return handler
}

func (c *PowerCommand) Dispatch([]byte) ([]byte, error) {
	rc := rpciOK

	log.Printf("dispatching power op %q", c.name)

	if c.Handler == nil {
		if c.state == powerStateHalt || c.state == powerStateReboot {
			rc = rpciERR
		}
	}

	msg := fmt.Sprintf("tools.os.statechange.status %s%d\x00", rc, c.state)

	if _, err := c.out.Request([]byte(msg)); err != nil {
		log.Printf("unable to send %q: %q", msg, err)
	}

	if c.Handler != nil {
		if err := c.Handler(); err != nil {
			log.Printf("%s: %s", c.name, err)
		}
	}

	return nil, nil
}

func Halt() error {
	log.Printf("Halting system...")
	// #nosec: Subprocess launching with variable
	return exec.Command(shutdown, "-h", "now").Run()
}

func Reboot() error {
	log.Printf("Rebooting system...")
	// #nosec: Subprocess launching with variable
	return exec.Command(shutdown, "-r", "now").Run()
}
