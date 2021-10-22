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
	"errors"
	"testing"
)

func TestPowerCommandHandler(t *testing.T) {
	shutdown = "/bin/echo"

	in := new(mockChannelIn)
	out := new(mockChannelOut)

	service := NewService(in, out)
	power := service.Power

	// cover nil Handler and out.Receive paths
	_, _ = power.Halt.Dispatch(nil)

	out.reply = append(out.reply, rpciOK, rpciOK)

	power.Halt.Handler = Halt
	power.Reboot.Handler = Reboot
	power.Suspend.Handler = func() error {
		return errors.New("an error")
	}

	commands := []PowerCommand{
		power.Halt,
		power.Reboot,
		power.Suspend,
	}

	for _, cmd := range commands {
		_, _ = cmd.Dispatch(nil)
	}
}
