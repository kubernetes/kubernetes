//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package heketitest

import (
	"testing"

	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/tests"
)

func TestNewHeketiMockTestServer(t *testing.T) {
	c := &HeketiMockTestServerConfig{
		Auth:     true,
		AdminKey: "admin",
		UserKey:  "user",
		Logging:  true,
	}

	h := NewHeketiMockTestServer(c)
	tests.Assert(t, h != nil)
	tests.Assert(t, h.Ts != nil)
	tests.Assert(t, h.DbFile != "")
	tests.Assert(t, h.App != nil)
	h.Close()

	h = NewHeketiMockTestServerDefault()
	tests.Assert(t, h != nil)
	tests.Assert(t, h.Ts != nil)
	tests.Assert(t, h.DbFile != "")
	tests.Assert(t, h.App != nil)
}

func TestHeketiMockTestServer(t *testing.T) {
	c := &HeketiMockTestServerConfig{
		Auth:     true,
		AdminKey: "admin",
		UserKey:  "user",
	}

	h := NewHeketiMockTestServer(c)
	defer h.Close()

	api := client.NewClient(h.URL(), "admin", "admin")
	tests.Assert(t, api != nil)

	cluster, err := api.ClusterCreate()
	tests.Assert(t, err == nil)
	tests.Assert(t, cluster != nil)
	tests.Assert(t, len(cluster.Nodes) == 0)
	tests.Assert(t, len(cluster.Volumes) == 0)

	info, err := api.ClusterInfo(cluster.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, info.Id == cluster.Id)
	tests.Assert(t, len(info.Nodes) == 0)
	tests.Assert(t, len(info.Volumes) == 0)
}
