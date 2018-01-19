/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package object

import (
	"fmt"
	"path"
	"strings"
)

// DatastorePath contains the components of a datastore path.
type DatastorePath struct {
	Datastore string
	Path      string
}

// FromString parses a datastore path.
// Returns true if the path could be parsed, false otherwise.
func (p *DatastorePath) FromString(s string) bool {
	if len(s) == 0 {
		return false
	}

	s = strings.TrimSpace(s)

	if !strings.HasPrefix(s, "[") {
		return false
	}

	s = s[1:]

	ix := strings.Index(s, "]")
	if ix < 0 {
		return false
	}

	p.Datastore = s[:ix]
	p.Path = strings.TrimSpace(s[ix+1:])

	return true
}

// String formats a datastore path.
func (p *DatastorePath) String() string {
	s := fmt.Sprintf("[%s]", p.Datastore)

	if p.Path == "" {
		return s
	}

	return strings.Join([]string{s, p.Path}, " ")
}

// IsVMDK returns true if Path has a ".vmdk" extension
func (p *DatastorePath) IsVMDK() bool {
	return path.Ext(p.Path) == ".vmdk"
}
