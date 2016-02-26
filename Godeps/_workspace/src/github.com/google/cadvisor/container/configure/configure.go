// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package configure

import (
	"fmt"

	"github.com/google/cadvisor/container/docker"
	"github.com/google/cadvisor/container/rkt"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"

	"github.com/golang/glog"
)

type ContainerConfiguration interface {
	GetFsContext() (c fs.Context, err error)
	RegisterRuntime(m info.MachineInfoFactory, f fs.FsInfo) error
}

func GetContConfigure(runtime string) (ContainerConfiguration, error) {
	glog.Infof("runtime = %q", runtime)
	switch runtime {
	case "docker":
		return dockerConfigure{}, nil
	case "rkt":
		return rktConfigure{}, nil
	default:
		return nil, fmt.Errorf("Unknown container runtime: %v", runtime)
	}
}

type dockerConfigure struct {
}

func (d dockerConfigure) GetFsContext() (fs.Context, error) {
	dockerInfo, err := docker.DockerInfo()
	if err != nil {
		return fs.Context{}, err
	}

	return fs.Context{DockerRoot: docker.RootDir(), DockerInfo: dockerInfo}, nil
}

func (d dockerConfigure) RegisterRuntime(m info.MachineInfoFactory, f fs.FsInfo) error {
	return docker.Register(m, f)
}

type rktConfigure struct {
}

func (r rktConfigure) GetFsContext() (fs.Context, error) {
	return fs.Context{RktPath: rkt.RktPath()}, nil
}

func (r rktConfigure) RegisterRuntime(m info.MachineInfoFactory, f fs.FsInfo) error {
//	return rkt.Register(m, f)
	return nil
}
