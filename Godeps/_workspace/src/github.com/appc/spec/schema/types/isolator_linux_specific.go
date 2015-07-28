// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
	"errors"
)

const (
	LinuxCapabilitiesRetainSetName = "os/linux/capabilities-retain-set"
	LinuxCapabilitiesRevokeSetName = "os/linux/capabilities-remove-set"
)

var LinuxIsolatorNames = make(map[ACIdentifier]struct{})

func init() {
	AddIsolatorName(LinuxCapabilitiesRetainSetName, LinuxIsolatorNames)
	AddIsolatorName(LinuxCapabilitiesRevokeSetName, LinuxIsolatorNames)

	AddIsolatorValueConstructor(LinuxCapabilitiesRetainSetName, NewLinuxCapabilitiesRetainSet)
	AddIsolatorValueConstructor(LinuxCapabilitiesRevokeSetName, NewLinuxCapabilitiesRevokeSet)
}

type LinuxCapabilitiesSet interface {
	Set() []LinuxCapability
	AssertValid() error
}

type LinuxCapability string
type linuxCapabilitiesSetValue struct {
	Set []LinuxCapability `json:"set"`
}

type linuxCapabilitiesSetBase struct {
	val linuxCapabilitiesSetValue
}

func (l linuxCapabilitiesSetBase) AssertValid() error {
	if len(l.val.Set) == 0 {
		return errors.New("set must be non-empty")
	}
	return nil
}

func (l *linuxCapabilitiesSetBase) UnmarshalJSON(b []byte) error {
	var v linuxCapabilitiesSetValue
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	l.val = v

	return err
}

func (l linuxCapabilitiesSetBase) Set() []LinuxCapability {
	return l.val.Set
}

func NewLinuxCapabilitiesRetainSet() IsolatorValue {
	return &LinuxCapabilitiesRetainSet{}
}

type LinuxCapabilitiesRetainSet struct {
	linuxCapabilitiesSetBase
}

func NewLinuxCapabilitiesRevokeSet() IsolatorValue {
	return &LinuxCapabilitiesRevokeSet{}
}

type LinuxCapabilitiesRevokeSet struct {
	linuxCapabilitiesSetBase
}
