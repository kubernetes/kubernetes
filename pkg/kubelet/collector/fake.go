// +build cgo,linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package collector

type Fake struct {
}

var _ Interface = new(Fake)

func (cc *Fake) Start() error {
	return nil
}

func (cc *Fake) MachineInfo() (*MachineInfo, error) {
	return &MachineInfo{}, nil
}

func (cc *Fake) VersionInfo() (*VersionInfo, error) {
	return &VersionInfo{}, nil
}

func (cc *Fake) FsInfo(fsLabel string) (*FsInfo, error) {
	return &FsInfo{}, nil
}

func (cc *Fake) WatchEvents(request *Request) (chan *Event, error) {
	return make(chan *Event), nil
}

func (cc *Fake) ContainerInfo(containerName string, req *ContainerInfoRequest, subcontainers bool, isRawContainer bool) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}
