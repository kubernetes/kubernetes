/*
Copyright 2022 The Kubernetes Authors.

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

package schema

import (
	"strconv"
	"strings"
)

// UnknownFieldPathOptions allow for tracking paths to unknown fields.
type UnknownFieldPathOptions struct {
	// TrackUnknownFieldPaths determines whether or not unknown field
	// paths should be stored or not.
	TrackUnknownFieldPaths bool
	// ParentPath builds the path to unknown fields as the object
	// is recursively traversed.
	ParentPath []string
	// UnknownFieldPaths is the list of all unknown fields identified.
	UnknownFieldPaths []string
}

// RecordUnknownFields adds a path to an unknown field to the
// record of UnknownFieldPaths, if TrackUnknownFieldPaths is true
func (o *UnknownFieldPathOptions) RecordUnknownField(field string) {
	if !o.TrackUnknownFieldPaths {
		return
	}
	l := len(o.ParentPath)
	o.AppendKey(field)
	o.UnknownFieldPaths = append(o.UnknownFieldPaths, strings.Join(o.ParentPath, ""))
	o.ParentPath = o.ParentPath[:l]
}

// AppendKey adds a key (i.e. field) to the current parent
// path, if TrackUnknownFieldPaths is true.
func (o *UnknownFieldPathOptions) AppendKey(key string) {
	if !o.TrackUnknownFieldPaths {
		return
	}
	if len(o.ParentPath) > 0 {
		o.ParentPath = append(o.ParentPath, ".")
	}
	o.ParentPath = append(o.ParentPath, key)
}

// AppendIndex adds an index to the most recent field of
// the current parent path, if TrackUnknownFieldPaths is true.
func (o *UnknownFieldPathOptions) AppendIndex(index int) {
	if !o.TrackUnknownFieldPaths {
		return
	}
	o.ParentPath = append(o.ParentPath, "[", strconv.Itoa(index), "]")
}
