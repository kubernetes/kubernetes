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
	"net/http"
	"sync"

	"k8s.io/apiserver/pkg/server/flagz"
	"k8s.io/utils/clock"
)

// healthMux is an interface describing the methods InstallHandler requires.
type flagzMux interface {
	Handle(pattern string, handler http.Handler)
}

type flagzRegistry struct {
	path           string
	lock           sync.Mutex
	flags          []flagz.Flag
	flagsInstalled bool
	clock          clock.Clock
}

func (reg *flagzRegistry) addFlag(flags ...flagz.Flag) error {
	return reg.addFlags(flags...)
}

func (reg *flagzRegistry) addFlags(flags ...flagz.Flag) error {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	if reg.flagsInstalled {
		return fmt.Errorf("unable to add because the %s endpoint has already been created", reg.path)
	}
	for _, flag := range flags {
		reg.flags = append(reg.flags, flag)
	}
	return nil
}

func (reg *flagzRegistry) installHandler(mux healthMux) {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	reg.flagsInstalled = true
	flagz.InstallHandler(mux, reg.flags...)
}

// AddFlags adds component flags to flagz endpoints.
func (s *GenericAPIServer) AddFlags(checks ...flagz.Flag) error {
	return s.addFlags(checks...)
}

// addFlags adds component flags to flagz endpoint.
func (s *GenericAPIServer) addFlags(flags ...flagz.Flag) error {
	return s.flagzRegistry.addFlags(flags...)
}

// installHealthz creates the healthz endpoint for this server
func (s *GenericAPIServer) installFlagz() {
	s.flagzRegistry.installHandler(s.Handler.NonGoRestfulMux)
}
