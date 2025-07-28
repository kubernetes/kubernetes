// Copyright 2019 The original author or authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package layout

import "path/filepath"

// Path represents an OCI image layout rooted in a file system path
type Path string

func (l Path) path(elem ...string) string {
	complete := []string{string(l)}
	return filepath.Join(append(complete, elem...)...)
}
