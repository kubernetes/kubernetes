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

package lmctfy

import (
	"errors"
	"log"
	"os/exec"

	"github.com/google/cadvisor/container"
)

func Register(paths ...string) error {
	if _, err := exec.LookPath("lmctfy"); err != nil {
		return errors.New("cannot find lmctfy")
	}
	f := &lmctfyFactory{}
	for _, path := range paths {
		log.Printf("register lmctfy under %v", path)
		container.RegisterContainerHandlerFactory(path, f)
	}
	return nil
}

type lmctfyFactory struct {
}

func (self *lmctfyFactory) String() string {
	return "lmctfy"
}

func (self *lmctfyFactory) NewContainerHandler(name string) (container.ContainerHandler, error) {
	c, err := New(name)
	if err != nil {
		return nil, err
	}
	// XXX(dengnan): /user is created by ubuntu 14.04. Not sure if we should list it
	handler := container.NewBlackListFilter(c, "/user")
	return handler, nil
}
