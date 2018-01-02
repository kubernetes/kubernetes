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
	"testing"

	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/tests"
)

func TestEntryStates(t *testing.T) {
	e := &Entry{}

	tests.Assert(t, e.State == api.EntryStateUnknown)
	tests.Assert(t, e.isOnline() == false)

	e.State = api.EntryStateOnline
	tests.Assert(t, e.isOnline())

	e.State = api.EntryStateOffline
	tests.Assert(t, e.isOnline() == false)

	e.State = api.EntryStateFailed
	tests.Assert(t, e.isOnline() == false)

}
