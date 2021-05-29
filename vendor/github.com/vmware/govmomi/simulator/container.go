/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"bytes"
	"encoding/json"
	"log"
	"os/exec"
	"strings"

	"github.com/vmware/govmomi/vim25/types"
)

// container provides methods to manage a container within a simulator VM lifecycle.
type container struct {
	id string
}

// inspect applies container network settings to vm.Guest properties.
func (c *container) inspect(vm *VirtualMachine) error {
	if c.id == "" {
		return nil
	}

	var objects []struct {
		NetworkSettings struct {
			Gateway     string
			IPAddress   string
			IPPrefixLen int
			MacAddress  string
		}
	}

	cmd := exec.Command("docker", "inspect", c.id)
	out, err := cmd.Output()
	if err != nil {
		return err
	}
	if err = json.NewDecoder(bytes.NewReader(out)).Decode(&objects); err != nil {
		return err
	}

	vm.Config.Annotation = strings.Join(cmd.Args, " ")
	vm.logPrintf("%s: %s", vm.Config.Annotation, string(out))

	for _, o := range objects {
		s := o.NetworkSettings
		if s.IPAddress == "" {
			continue
		}

		vm.Guest.IpAddress = s.IPAddress
		vm.Summary.Guest.IpAddress = s.IPAddress

		if len(vm.Guest.Net) != 0 {
			net := &vm.Guest.Net[0]
			net.IpAddress = []string{s.IPAddress}
			net.MacAddress = s.MacAddress
		}
	}

	return nil
}

// start runs the container if specified by the RUN.container extraConfig property.
func (c *container) start(vm *VirtualMachine) {
	if c.id != "" {
		start := "start"
		if vm.Runtime.PowerState == types.VirtualMachinePowerStateSuspended {
			start = "unpause"
		}
		cmd := exec.Command("docker", start, c.id)
		err := cmd.Run()
		if err != nil {
			log.Printf("%s %s: %s", vm.Name, cmd.Args, err)
		}
		return
	}

	var args []string

	for _, opt := range vm.Config.ExtraConfig {
		val := opt.GetOptionValue()
		if val.Key == "RUN.container" {
			run := val.Value.(string)
			err := json.Unmarshal([]byte(run), &args)
			if err != nil {
				args = []string{run}
			}

			break
		}
	}

	if len(args) == 0 {
		return
	}

	args = append([]string{"run", "-d", "--name", vm.Name}, args...)
	cmd := exec.Command("docker", args...)
	out, err := cmd.Output()
	if err != nil {
		log.Printf("%s %s: %s", vm.Name, cmd.Args, err)
		return
	}

	c.id = strings.TrimSpace(string(out))
	vm.logPrintf("%s %s: %s", cmd.Path, cmd.Args, c.id)

	if err = c.inspect(vm); err != nil {
		log.Printf("%s inspect %s: %s", vm.Name, c.id, err)
	}
}

// stop the container (if any) for the given vm.
func (c *container) stop(vm *VirtualMachine) {
	if c.id == "" {
		return
	}

	cmd := exec.Command("docker", "stop", c.id)
	err := cmd.Run()
	if err != nil {
		log.Printf("%s %s: %s", vm.Name, cmd.Args, err)
	}
}

// pause the container (if any) for the given vm.
func (c *container) pause(vm *VirtualMachine) {
	if c.id == "" {
		return
	}

	cmd := exec.Command("docker", "pause", c.id)
	err := cmd.Run()
	if err != nil {
		log.Printf("%s %s: %s", vm.Name, cmd.Args, err)
	}
}

// remove the container (if any) for the given vm.
func (c *container) remove(vm *VirtualMachine) {
	if c.id == "" {
		return
	}

	cmd := exec.Command("docker", "rm", "-f", c.id)
	err := cmd.Run()
	if err != nil {
		log.Printf("%s %s: %s", vm.Name, cmd.Args, err)
	}
}
