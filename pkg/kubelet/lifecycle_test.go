/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"testing"
)

type commandState struct {
	id      string
	command []string
}

type fakeCommandRunner struct {
	commands []commandState
	output   []byte
	err      error
}

func (f *fakeCommandRunner) RunInContainer(id string, cmd []string) ([]byte, error) {
	f.commands = append(f.commands, commandState{id: id, command: cmd})
	return f.output, f.err
}

func TestOnStart(t *testing.T) {
	fake := fakeCommandRunner{}
	lifecycle := newCommandLineLifecycle(&fake)
	lifecycle.OnStart("foo")

	if len(fake.commands) != 1 || fake.commands[0].id != "foo" || fake.commands[0].command[0] != "start.sh" {
		t.Errorf("unexpected commands: %v", fake.commands)
	}
}

func TestOnStop(t *testing.T) {
	fake := fakeCommandRunner{}
	lifecycle := newCommandLineLifecycle(&fake)
	lifecycle.OnStop("foo")

	if len(fake.commands) != 1 || fake.commands[0].id != "foo" || fake.commands[0].command[0] != "stop.sh" {
		t.Errorf("unexpected commands: %v", fake.commands)
	}
}
