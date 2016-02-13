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

package rkt

import (
	"flag"
	"fmt"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
)

// Basepath to all container specific information that libcontainer stores.
var rktPath = flag.String("rkt_path", "/var/lib/rkt", "Absolute path to the Rkt root directory")

// The namespace under which Docker aliases are unique.
var RktNamespace = "rkt"

func RktPath() string {
	return *rktPath
}

type rktFactory struct {
	machineInfoFactory info.MachineInfoFactory

	// Information about mounted filesystems.
	fsInfo fs.FsInfo
}

func (self *rktFactory) String() string {
	return RktNamespace
}

func (self *rktFactory) NewContainerHandler(name string, inHostNamespace bool) (handler container.ContainerHandler, err error) {
	return nil, fmt.Errorf("Unsupported")
}

func (self *rktFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	return false, false, fmt.Errorf("Unsupported")
}

func (self *rktFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

func Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo) error {
	f := &rktFactory{
		machineInfoFactory: factory,
		fsInfo:             fsInfo,
	}
	container.RegisterContainerHandlerFactory(f)

	return nil
}
