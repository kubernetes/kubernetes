// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package module provides functions for interacting with modules.

The appengine package contains functions that report the identity of the app,
including the module name.
*/
package module

import (
	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	pb "google.golang.org/appengine/internal/modules"
)

// List returns the names of modules belonging to this application.
func List(c appengine.Context) ([]string, error) {
	req := &pb.GetModulesRequest{}
	res := &pb.GetModulesResponse{}
	err := c.Call("modules", "GetModules", req, res, nil)
	return res.Module, err
}

// NumInstances returns the number of instances of the given module/version.
// If either argument is the empty string it means the default.
func NumInstances(c appengine.Context, module, version string) (int, error) {
	req := &pb.GetNumInstancesRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.GetNumInstancesResponse{}

	if err := c.Call("modules", "GetNumInstances", req, res, nil); err != nil {
		return 0, err
	}
	return int(*res.Instances), nil
}

// SetNumInstances sets the number of instances of the given module.version to the
// specified value. If either module or version are the empty string it means the
// default.
func SetNumInstances(c appengine.Context, module, version string, instances int) error {
	req := &pb.SetNumInstancesRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	req.Instances = proto.Int64(int64(instances))
	res := &pb.SetNumInstancesResponse{}
	return c.Call("modules", "SetNumInstances", req, res, nil)
}

// Versions returns the names of the versions that belong to the specified module.
// If module is the empty string, it means the default module.
func Versions(c appengine.Context, module string) ([]string, error) {
	req := &pb.GetVersionsRequest{}
	if module != "" {
		req.Module = &module
	}
	res := &pb.GetVersionsResponse{}
	err := c.Call("modules", "GetVersions", req, res, nil)
	return res.GetVersion(), err
}

// DefaultVersion returns the default version of the specified module.
// If module is the empty string, it means the default module.
func DefaultVersion(c appengine.Context, module string) (string, error) {
	req := &pb.GetDefaultVersionRequest{}
	if module != "" {
		req.Module = &module
	}
	res := &pb.GetDefaultVersionResponse{}
	err := c.Call("modules", "GetDefaultVersion", req, res, nil)
	return res.GetVersion(), err
}

// Start starts the specified version of the specified module.
// If either module or version are the empty string, it means the default.
func Start(c appengine.Context, module, version string) error {
	req := &pb.StartModuleRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.StartModuleResponse{}
	return c.Call("modules", "StartModule", req, res, nil)
}

// Stop stops the specified version of the specified module.
// If either module or version are the empty string, it means the default.
func Stop(c appengine.Context, module, version string) error {
	req := &pb.StopModuleRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.StopModuleResponse{}
	return c.Call("modules", "StopModule", req, res, nil)
}
