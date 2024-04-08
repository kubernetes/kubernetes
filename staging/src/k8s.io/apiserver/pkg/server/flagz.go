/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package server

import (
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/server/flagz"
	cliflag "k8s.io/component-base/cli/flag"
)

type flagzRegistry struct {
	path           string
	lock           sync.Mutex
	flags          []cliflag.NamedFlagSets
	flagsInstalled bool
}

// AddFlags adds component flags to flagz endpoints.
func (s *GenericAPIServer) AddFlags(flags cliflag.NamedFlagSets) error {
	return s.flagzRegistry.addFlags(flags)
}

func (reg *flagzRegistry) addFlags(flags cliflag.NamedFlagSets) error {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	if reg.flagsInstalled {
		return fmt.Errorf("unable to add because the %s endpoint has already been created", reg.path)
	}
	reg.flags = append(reg.flags, flags)
	return nil
}

// installFlagz creates the flagz endpoint for this server
func (s *GenericAPIServer) installFlagz() {
	s.flagzRegistry.installHandler(s.Handler.NonGoRestfulMux)
}

func (reg *flagzRegistry) installHandler(mux healthMux) {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	reg.flagsInstalled = true
	flagz.InstallHandler(mux, reg.flags...)
}
