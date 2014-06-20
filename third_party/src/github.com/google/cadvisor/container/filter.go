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

package container

import (
	"strings"

	"github.com/google/cadvisor/info"
)

type containerListFilter struct {
	filter  func(string) bool
	handler ContainerHandler
}

func (self *containerListFilter) ContainerReference() (info.ContainerReference, error) {
	return self.handler.ContainerReference()
}

func (self *containerListFilter) GetSpec() (*info.ContainerSpec, error) {
	return self.handler.GetSpec()
}

func (self *containerListFilter) GetStats() (*info.ContainerStats, error) {
	return self.handler.GetStats()
}

func (self *containerListFilter) ListContainers(listType ListType) ([]info.ContainerReference, error) {
	containers, err := self.handler.ListContainers(listType)
	if err != nil {
		return nil, err
	}
	if len(containers) == 0 {
		return nil, nil
	}
	ret := make([]info.ContainerReference, 0, len(containers))
	for _, c := range containers {
		if self.filter(c.Name) {
			ret = append(ret, c)
		}
	}
	return ret, nil
}

func (self *containerListFilter) ListThreads(listType ListType) ([]int, error) {
	return self.handler.ListThreads(listType)
}

func (self *containerListFilter) ListProcesses(listType ListType) ([]int, error) {
	return self.handler.ListProcesses(listType)
}

func NewWhiteListFilter(handler ContainerHandler, acceptedPaths ...string) ContainerHandler {
	filter := func(p string) bool {
		for _, path := range acceptedPaths {
			if strings.HasPrefix(p, path) {
				return true
			}
		}
		return false
	}
	return &containerListFilter{
		filter:  filter,
		handler: handler,
	}
}

func NewBlackListFilter(handler ContainerHandler, forbiddenPaths ...string) ContainerHandler {
	filter := func(p string) bool {
		for _, path := range forbiddenPaths {
			if strings.HasPrefix(p, path) {
				return false
			}
		}
		return true
	}
	return &containerListFilter{
		filter:  filter,
		handler: handler,
	}
}
