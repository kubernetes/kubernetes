// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package module

import (
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/modules"
)

const version = "test-version"
const module = "test-module"
const instances = 3

func TestList(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "GetModules", func(req *pb.GetModulesRequest, res *pb.GetModulesResponse) error {
		res.Module = []string{"default", "mod1"}
		return nil
	})
	got, err := List(c)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	want := []string{"default", "mod1"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("List = %v, want %v", got, want)
	}
}

func TestSetNumInstances(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "SetNumInstances", func(req *pb.SetNumInstancesRequest, res *pb.SetNumInstancesResponse) error {
		if *req.Module != module {
			t.Errorf("Module = %v, want %v", req.Module, module)
		}
		if *req.Version != version {
			t.Errorf("Version = %v, want %v", req.Version, version)
		}
		if *req.Instances != instances {
			t.Errorf("Instances = %v, want %d", req.Instances, instances)
		}
		return nil
	})
	err := SetNumInstances(c, module, version, instances)
	if err != nil {
		t.Fatalf("SetNumInstances: %v", err)
	}
}

func TestVersions(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "GetVersions", func(req *pb.GetVersionsRequest, res *pb.GetVersionsResponse) error {
		if *req.Module != module {
			t.Errorf("Module = %v, want %v", req.Module, module)
		}
		res.Version = []string{"v1", "v2", "v3"}
		return nil
	})
	got, err := Versions(c, module)
	if err != nil {
		t.Fatalf("Versions: %v", err)
	}
	want := []string{"v1", "v2", "v3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Versions = %v, want %v", got, want)
	}
}

func TestDefaultVersion(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "GetDefaultVersion", func(req *pb.GetDefaultVersionRequest, res *pb.GetDefaultVersionResponse) error {
		if *req.Module != module {
			t.Errorf("Module = %v, want %v", req.Module, module)
		}
		res.Version = proto.String(version)
		return nil
	})
	got, err := DefaultVersion(c, module)
	if err != nil {
		t.Fatalf("DefaultVersion: %v", err)
	}
	if got != version {
		t.Errorf("Version = %v, want %v", got, version)
	}
}

func TestStart(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "StartModule", func(req *pb.StartModuleRequest, res *pb.StartModuleResponse) error {
		if *req.Module != module {
			t.Errorf("Module = %v, want %v", req.Module, module)
		}
		if *req.Version != version {
			t.Errorf("Version = %v, want %v", req.Version, version)
		}
		return nil
	})

	err := Start(c, module, version)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
}

func TestStop(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "modules", "StopModule", func(req *pb.StopModuleRequest, res *pb.StopModuleResponse) error {
		version := "test-version"
		module := "test-module"
		if *req.Module != module {
			t.Errorf("Module = %v, want %v", req.Module, module)
		}
		if *req.Version != version {
			t.Errorf("Version = %v, want %v", req.Version, version)
		}
		return nil
	})

	err := Stop(c, module, version)
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}
}
