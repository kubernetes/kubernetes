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

import "path/filepath"

func NewNamed(root string, name Name) *namedController {
	return &namedController{
		root: root,
		name: name,
	}
}

type namedController struct {
	root string
	name Name
}

func (n *namedController) Name() Name {
	return n.name
}

func (n *namedController) Path(path string) string {
	return filepath.Join(n.root, string(n.name), path)
}
