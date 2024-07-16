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

package cgroup1

import (
	"fmt"
	"os"
	"path/filepath"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func NewNetPrio(root string) *netprioController {
	return &netprioController{
		root: filepath.Join(root, string(NetPrio)),
	}
}

type netprioController struct {
	root string
}

func (n *netprioController) Name() Name {
	return NetPrio
}

func (n *netprioController) Path(path string) string {
	return filepath.Join(n.root, path)
}

func (n *netprioController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(n.Path(path), defaultDirPerm); err != nil {
		return err
	}
	if resources.Network != nil {
		for _, prio := range resources.Network.Priorities {
			if err := os.WriteFile(
				filepath.Join(n.Path(path), "net_prio.ifpriomap"),
				formatPrio(prio.Name, prio.Priority),
				defaultFilePerm,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func formatPrio(name string, prio uint32) []byte {
	return []byte(fmt.Sprintf("%s %d", name, prio))
}
