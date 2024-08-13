/*
   Copyright The containerd Authors.

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

package cgroups

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

func NewFreezer(root string) *freezerController {
	return &freezerController{
		root: filepath.Join(root, string(Freezer)),
	}
}

type freezerController struct {
	root string
}

func (f *freezerController) Name() Name {
	return Freezer
}

func (f *freezerController) Path(path string) string {
	return filepath.Join(f.root, path)
}

func (f *freezerController) Freeze(path string) error {
	return f.waitState(path, Frozen)
}

func (f *freezerController) Thaw(path string) error {
	return f.waitState(path, Thawed)
}

func (f *freezerController) changeState(path string, state State) error {
	return retryingWriteFile(
		filepath.Join(f.root, path, "freezer.state"),
		[]byte(strings.ToUpper(string(state))),
		defaultFilePerm,
	)
}

func (f *freezerController) state(path string) (State, error) {
	current, err := os.ReadFile(filepath.Join(f.root, path, "freezer.state"))
	if err != nil {
		return "", err
	}
	return State(strings.ToLower(strings.TrimSpace(string(current)))), nil
}

func (f *freezerController) waitState(path string, state State) error {
	for {
		if err := f.changeState(path, state); err != nil {
			return err
		}
		current, err := f.state(path)
		if err != nil {
			return err
		}
		if current == state {
			return nil
		}
		time.Sleep(1 * time.Millisecond)
	}
}
