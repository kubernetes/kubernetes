//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"github.com/heketi/heketi/pkg/glusterfs/api"
)

type Entry struct {
	State api.EntryState
}

func (e *Entry) isOnline() bool {
	return e.State == api.EntryStateOnline
}

func (e *Entry) SetOnline() {
	e.State = api.EntryStateOnline
}
