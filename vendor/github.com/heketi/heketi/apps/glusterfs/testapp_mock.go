//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"bytes"
	"github.com/lpabon/godbc"
)

func NewTestApp(dbfile string) *App {

	// Create simple configuration for unit tests
	appConfig := bytes.NewBuffer([]byte(`{
		"glusterfs" : {
			"executor" : "mock",
			"allocator" : "simple",
			"db" : "` + dbfile + `"
		}
	}`))
	app := NewApp(appConfig)
	godbc.Check(app != nil)

	return app
}
