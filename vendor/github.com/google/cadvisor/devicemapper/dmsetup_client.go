// Copyright 2016 Google Inc. All Rights Reserved.
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
package devicemapper

import (
	"os/exec"
	"strconv"
	"strings"

	"github.com/golang/glog"
)

// DmsetupClient is a low-level client for interacting with devicemapper via
// the dmsetup utility.
type DmsetupClient interface {
	Table(deviceName string) ([]byte, error)
	Message(deviceName string, sector int, message string) ([]byte, error)
	Status(deviceName string) ([]byte, error)
}

func NewDmsetupClient() DmsetupClient {
	return &defaultDmsetupClient{}
}

// defaultDmsetupClient implements the standard behavior for interacting with dmsetup.
type defaultDmsetupClient struct{}

var _ DmsetupClient = &defaultDmsetupClient{}

func (c *defaultDmsetupClient) Table(deviceName string) ([]byte, error) {
	return c.dmsetup("table", deviceName)
}

func (c *defaultDmsetupClient) Message(deviceName string, sector int, message string) ([]byte, error) {
	return c.dmsetup("message", deviceName, strconv.Itoa(sector), message)
}

func (c *defaultDmsetupClient) Status(deviceName string) ([]byte, error) {
	return c.dmsetup("status", deviceName)
}

func (*defaultDmsetupClient) dmsetup(args ...string) ([]byte, error) {
	glog.V(5).Infof("running dmsetup %v", strings.Join(args, " "))
	return exec.Command("dmsetup", args...).Output()
}
