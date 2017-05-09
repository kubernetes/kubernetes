/*
Copyright 2017 The Kubernetes Authors.

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

package cm

import (
	"sync"
	"github.com/golang/glog"
)

var CPUManagerSingleton CPUManager
var once sync.Once


type CPUManager interface {
	Start ()
}

type cpuManagerImpl struct {
}

var _ CPUManager = &cpuManagerImpl{}

func GetCPUManagerSingleton(enabled bool) CPUManager {
	once.Do(func() {
		if enabled {
			CPUManagerSingleton = &cpuManagerImpl{}
		} else {
			CPUManagerSingleton = &cpuManagerNoop{}
		}
	})
	return CPUManagerSingleton
}

func (cmg *cpuManagerImpl) Start() {
	glog.V(3).Infof("Starting CPU Manager")
	return
}

type cpuManagerNoop struct {
}

var _ CPUManager = &cpuManagerNoop{}

func (cmg *cpuManagerNoop ) Start() {
	return
}