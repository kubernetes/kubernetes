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
	"strconv"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func NewNetCls(root string) *netclsController {
	return &netclsController{
		root: filepath.Join(root, string(NetCLS)),
	}
}

type netclsController struct {
	root string
}

func (n *netclsController) Name() Name {
	return NetCLS
}

func (n *netclsController) Path(path string) string {
	return filepath.Join(n.root, path)
}

func (n *netclsController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(n.Path(path), defaultDirPerm); err != nil {
		return err
	}
	if resources.Network != nil && resources.Network.ClassID != nil && *resources.Network.ClassID > 0 {
		return retryingWriteFile(
			filepath.Join(n.Path(path), "net_cls.classid"),
			[]byte(strconv.FormatUint(uint64(*resources.Network.ClassID), 10)),
			defaultFilePerm,
		)
	}
	return nil
}

func (n *netclsController) Update(path string, resources *specs.LinuxResources) error {
	return n.Create(path, resources)
}
