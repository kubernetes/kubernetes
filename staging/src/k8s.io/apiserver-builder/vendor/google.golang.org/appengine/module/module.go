// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package module provides functions for interacting with modules.

The appengine package contains functions that report the identity of the app,
including the module name.
*/
package module // import "google.golang.org/appengine/module"

import (
	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/modules"
)

// List returns the names of modules belonging to this application.
func List(c context.Context) ([]string, error) {
	req := &pb.GetModulesRequest{}
	res := &pb.GetModulesResponse{}
	err := internal.Call(c, "modules", "GetModules", req, res)
	return res.Module, err
}

// NumInstances returns the number of instances of the given module/version.
// If either argument is the empty string it means the default.
func NumInstances(c context.Context, module, version string) (int, error) {
	req := &pb.GetNumInstancesRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.GetNumInstancesResponse{}

	if err := internal.Call(c, "modules", "GetNumInstances", req, res); err != nil {
		return 0, err
	}
	return int(*res.Instances), nil
}

// SetNumInstances sets the number of instances of the given module.version to the
// specified value. If either module or version are the empty string it means the
// default.
func SetNumInstances(c context.Context, module, version string, instances int) error {
	req := &pb.SetNumInstancesRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	req.Instances = proto.Int64(int64(instances))
	res := &pb.SetNumInstancesResponse{}
	return internal.Call(c, "modules", "SetNumInstances", req, res)
}

// Versions returns the names of the versions that belong to the specified module.
// If module is the empty string, it means the default module.
func Versions(c context.Context, module string) ([]string, error) {
	req := &pb.GetVersionsRequest{}
	if module != "" {
		req.Module = &module
	}
	res := &pb.GetVersionsResponse{}
	err := internal.Call(c, "modules", "GetVersions", req, res)
	return res.GetVersion(), err
}

// DefaultVersion returns the default version of the specified module.
// If module is the empty string, it means the default module.
func DefaultVersion(c context.Context, module string) (string, error) {
	req := &pb.GetDefaultVersionRequest{}
	if module != "" {
		req.Module = &module
	}
	res := &pb.GetDefaultVersionResponse{}
	err := internal.Call(c, "modules", "GetDefaultVersion", req, res)
	return res.GetVersion(), err
}

// Start starts the specified version of the specified module.
// If either module or version are the empty string, it means the default.
func Start(c context.Context, module, version string) error {
	req := &pb.StartModuleRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.StartModuleResponse{}
	return internal.Call(c, "modules", "StartModule", req, res)
}

// Stop stops the specified version of the specified module.
// If either module or version are the empty string, it means the default.
func Stop(c context.Context, module, version string) error {
	req := &pb.StopModuleRequest{}
	if module != "" {
		req.Module = &module
	}
	if version != "" {
		req.Version = &version
	}
	res := &pb.StopModuleResponse{}
	return internal.Call(c, "modules", "StopModule", req, res)
}
