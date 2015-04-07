// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import "fmt"

// ChangeType is a type for constants indicating the type of change
// in a container
type ChangeType int

const (
	// ChangeModify is the ChangeType for container modifications
	ChangeModify ChangeType = iota

	// ChangeAdd is the ChangeType for additions to a container
	ChangeAdd

	// ChangeDelete is the ChangeType for deletions from a container
	ChangeDelete
)

// Change represents a change in a container.
//
// See http://goo.gl/QkW9sH for more details.
type Change struct {
	Path string
	Kind ChangeType
}

func (change *Change) String() string {
	var kind string
	switch change.Kind {
	case ChangeModify:
		kind = "C"
	case ChangeAdd:
		kind = "A"
	case ChangeDelete:
		kind = "D"
	}
	return fmt.Sprintf("%s %s", kind, change.Path)
}
