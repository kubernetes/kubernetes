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
	"fmt"
	"strings"
	"unicode"
)

const (
	LinuxCapabilitiesRetainSetName = "os/linux/capabilities-retain-set"
	LinuxCapabilitiesRevokeSetName = "os/linux/capabilities-remove-set"
	LinuxNoNewPrivilegesName       = "os/linux/no-new-privileges"
	LinuxSeccompRemoveSetName      = "os/linux/seccomp-remove-set"
	LinuxSeccompRetainSetName      = "os/linux/seccomp-retain-set"
	LinuxOOMScoreAdjName           = "os/linux/oom-score-adj"
	LinuxCPUSharesName             = "os/linux/cpu-shares"
	LinuxSELinuxContextName        = "os/linux/selinux-context"
)

var LinuxIsolatorNames = make(map[ACIdentifier]struct{})

func init() {
	for name, con := range map[ACIdentifier]IsolatorValueConstructor{
		LinuxCapabilitiesRevokeSetName: func() IsolatorValue { return &LinuxCapabilitiesRevokeSet{} },
		LinuxCapabilitiesRetainSetName: func() IsolatorValue { return &LinuxCapabilitiesRetainSet{} },
		LinuxNoNewPrivilegesName:       func() IsolatorValue { v := LinuxNoNewPrivileges(false); return &v },
		LinuxOOMScoreAdjName:           func() IsolatorValue { v := LinuxOOMScoreAdj(0); return &v },
		LinuxCPUSharesName:             func() IsolatorValue { v := LinuxCPUShares(1024); return &v },
		LinuxSeccompRemoveSetName:      func() IsolatorValue { return &LinuxSeccompRemoveSet{} },
		LinuxSeccompRetainSetName:      func() IsolatorValue { return &LinuxSeccompRetainSet{} },
		LinuxSELinuxContextName:        func() IsolatorValue { return &LinuxSELinuxContext{} },
	} {
		AddIsolatorName(name, LinuxIsolatorNames)
		AddIsolatorValueConstructor(name, con)
	}
}

type LinuxNoNewPrivileges bool

func (l LinuxNoNewPrivileges) AssertValid() error {
	return nil
}

// TODO(lucab): both need to be clarified in spec,
// see https://github.com/appc/spec/issues/625
func (l LinuxNoNewPrivileges) multipleAllowed() bool {
	return true
}
func (l LinuxNoNewPrivileges) Conflicts() []ACIdentifier {
	return nil
}

func (l *LinuxNoNewPrivileges) UnmarshalJSON(b []byte) error {
	var v bool
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	*l = LinuxNoNewPrivileges(v)

	return nil
}

type AsIsolator interface {
	AsIsolator() (*Isolator, error)
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

// TODO(lucab): both need to be clarified in spec,
// see https://github.com/appc/spec/issues/625
func (l linuxCapabilitiesSetBase) multipleAllowed() bool {
	return true
}
func (l linuxCapabilitiesSetBase) Conflicts() []ACIdentifier {
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

type LinuxCapabilitiesRetainSet struct {
	linuxCapabilitiesSetBase
}

func NewLinuxCapabilitiesRetainSet(caps ...string) (*LinuxCapabilitiesRetainSet, error) {
	l := LinuxCapabilitiesRetainSet{
		linuxCapabilitiesSetBase{
			linuxCapabilitiesSetValue{
				make([]LinuxCapability, len(caps)),
			},
		},
	}
	for i, c := range caps {
		l.linuxCapabilitiesSetBase.val.Set[i] = LinuxCapability(c)
	}
	if err := l.AssertValid(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l LinuxCapabilitiesRetainSet) AsIsolator() (*Isolator, error) {
	b, err := json.Marshal(l.linuxCapabilitiesSetBase.val)
	if err != nil {
		return nil, err
	}
	rm := json.RawMessage(b)
	return &Isolator{
		Name:     LinuxCapabilitiesRetainSetName,
		ValueRaw: &rm,
		value:    &l,
	}, nil
}

type LinuxCapabilitiesRevokeSet struct {
	linuxCapabilitiesSetBase
}

func NewLinuxCapabilitiesRevokeSet(caps ...string) (*LinuxCapabilitiesRevokeSet, error) {
	l := LinuxCapabilitiesRevokeSet{
		linuxCapabilitiesSetBase{
			linuxCapabilitiesSetValue{
				make([]LinuxCapability, len(caps)),
			},
		},
	}
	for i, c := range caps {
		l.linuxCapabilitiesSetBase.val.Set[i] = LinuxCapability(c)
	}
	if err := l.AssertValid(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l LinuxCapabilitiesRevokeSet) AsIsolator() (*Isolator, error) {
	b, err := json.Marshal(l.linuxCapabilitiesSetBase.val)
	if err != nil {
		return nil, err
	}
	rm := json.RawMessage(b)
	return &Isolator{
		Name:     LinuxCapabilitiesRevokeSetName,
		ValueRaw: &rm,
		value:    &l,
	}, nil
}

type LinuxSeccompSet interface {
	Set() []LinuxSeccompEntry
	Errno() LinuxSeccompErrno
	AssertValid() error
}

type LinuxSeccompEntry string
type LinuxSeccompErrno string

type linuxSeccompValue struct {
	Set   []LinuxSeccompEntry `json:"set"`
	Errno LinuxSeccompErrno   `json:"errno"`
}

type linuxSeccompBase struct {
	val linuxSeccompValue
}

func (l linuxSeccompBase) multipleAllowed() bool {
	return false
}

func (l linuxSeccompBase) AssertValid() error {
	if len(l.val.Set) == 0 {
		return errors.New("set must be non-empty")
	}
	if l.val.Errno == "" {
		return nil
	}
	for _, c := range l.val.Errno {
		if !unicode.IsUpper(c) {
			return errors.New("errno must be an upper case string")
		}
	}
	return nil
}

func (l *linuxSeccompBase) UnmarshalJSON(b []byte) error {
	var v linuxSeccompValue
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	l.val = v
	return nil
}

func (l linuxSeccompBase) Set() []LinuxSeccompEntry {
	return l.val.Set
}

func (l linuxSeccompBase) Errno() LinuxSeccompErrno {
	return l.val.Errno
}

type LinuxSeccompRetainSet struct {
	linuxSeccompBase
}

func (l LinuxSeccompRetainSet) Conflicts() []ACIdentifier {
	return []ACIdentifier{LinuxSeccompRemoveSetName}
}

func NewLinuxSeccompRetainSet(errno string, syscall ...string) (*LinuxSeccompRetainSet, error) {
	l := LinuxSeccompRetainSet{
		linuxSeccompBase{
			linuxSeccompValue{
				make([]LinuxSeccompEntry, len(syscall)),
				LinuxSeccompErrno(errno),
			},
		},
	}
	for i, c := range syscall {
		l.linuxSeccompBase.val.Set[i] = LinuxSeccompEntry(c)
	}
	if err := l.AssertValid(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l LinuxSeccompRetainSet) AsIsolator() (*Isolator, error) {
	b, err := json.Marshal(l.linuxSeccompBase.val)
	if err != nil {
		return nil, err
	}
	rm := json.RawMessage(b)
	return &Isolator{
		Name:     LinuxSeccompRetainSetName,
		ValueRaw: &rm,
		value:    &l,
	}, nil
}

type LinuxSeccompRemoveSet struct {
	linuxSeccompBase
}

func (l LinuxSeccompRemoveSet) Conflicts() []ACIdentifier {
	return []ACIdentifier{LinuxSeccompRetainSetName}
}

func NewLinuxSeccompRemoveSet(errno string, syscall ...string) (*LinuxSeccompRemoveSet, error) {
	l := LinuxSeccompRemoveSet{
		linuxSeccompBase{
			linuxSeccompValue{
				make([]LinuxSeccompEntry, len(syscall)),
				LinuxSeccompErrno(errno),
			},
		},
	}
	for i, c := range syscall {
		l.linuxSeccompBase.val.Set[i] = LinuxSeccompEntry(c)
	}
	if err := l.AssertValid(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l LinuxSeccompRemoveSet) AsIsolator() (*Isolator, error) {
	b, err := json.Marshal(l.linuxSeccompBase.val)
	if err != nil {
		return nil, err
	}
	rm := json.RawMessage(b)
	return &Isolator{
		Name:     LinuxSeccompRemoveSetName,
		ValueRaw: &rm,
		value:    &l,
	}, nil
}

// LinuxCPUShares assigns the CPU time share weight to the processes executed.
// See https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#CPUShares=weight,
// https://www.kernel.org/doc/Documentation/scheduler/sched-design-CFS.txt
type LinuxCPUShares int

func NewLinuxCPUShares(val int) (*LinuxCPUShares, error) {
	l := LinuxCPUShares(val)
	if err := l.AssertValid(); err != nil {
		return nil, err
	}

	return &l, nil
}

func (l LinuxCPUShares) AssertValid() error {
	if l < 2 || l > 262144 {
		return fmt.Errorf("%s must be between 2 and 262144, got %d", LinuxCPUSharesName, l)
	}
	return nil
}

func (l LinuxCPUShares) multipleAllowed() bool {
	return false
}

func (l LinuxCPUShares) Conflicts() []ACIdentifier {
	return nil
}

func (l *LinuxCPUShares) UnmarshalJSON(b []byte) error {
	var v int
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	*l = LinuxCPUShares(v)
	return nil
}

func (l LinuxCPUShares) AsIsolator() Isolator {
	b, err := json.Marshal(l)
	if err != nil {
		panic(err)
	}
	rm := json.RawMessage(b)
	return Isolator{
		Name:     LinuxCPUSharesName,
		ValueRaw: &rm,
		value:    &l,
	}
}

// LinuxOOMScoreAdj is equivalent to /proc/[pid]/oom_score_adj
type LinuxOOMScoreAdj int // -1000 to 1000

func NewLinuxOOMScoreAdj(val int) (*LinuxOOMScoreAdj, error) {
	l := LinuxOOMScoreAdj(val)
	if err := l.AssertValid(); err != nil {
		return nil, err
	}

	return &l, nil
}

func (l LinuxOOMScoreAdj) AssertValid() error {
	if l < -1000 || l > 1000 {
		return fmt.Errorf("%s must be between -1000 and 1000, got %d", LinuxOOMScoreAdjName, l)
	}
	return nil
}

func (l LinuxOOMScoreAdj) multipleAllowed() bool {
	return false
}

func (l LinuxOOMScoreAdj) Conflicts() []ACIdentifier {
	return nil
}

func (l *LinuxOOMScoreAdj) UnmarshalJSON(b []byte) error {
	var v int
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	*l = LinuxOOMScoreAdj(v)
	return nil
}

func (l LinuxOOMScoreAdj) AsIsolator() Isolator {
	b, err := json.Marshal(l)
	if err != nil {
		panic(err)
	}
	rm := json.RawMessage(b)
	return Isolator{
		Name:     LinuxOOMScoreAdjName,
		ValueRaw: &rm,
		value:    &l,
	}
}

type LinuxSELinuxUser string
type LinuxSELinuxRole string
type LinuxSELinuxType string
type LinuxSELinuxLevel string

type linuxSELinuxValue struct {
	User  LinuxSELinuxUser  `json:"user"`
	Role  LinuxSELinuxRole  `json:"role"`
	Type  LinuxSELinuxType  `json:"type"`
	Level LinuxSELinuxLevel `json:"level"`
}

type LinuxSELinuxContext struct {
	val linuxSELinuxValue
}

func (l LinuxSELinuxContext) AssertValid() error {
	if l.val.User == "" || strings.Contains(string(l.val.User), ":") {
		return fmt.Errorf("invalid user value %q", l.val.User)
	}
	if l.val.Role == "" || strings.Contains(string(l.val.Role), ":") {
		return fmt.Errorf("invalid role value %q", l.val.Role)
	}
	if l.val.Type == "" || strings.Contains(string(l.val.Type), ":") {
		return fmt.Errorf("invalid type value %q", l.val.Type)
	}
	if l.val.Level == "" {
		return fmt.Errorf("invalid level value %q", l.val.Level)
	}
	return nil
}

func (l *LinuxSELinuxContext) UnmarshalJSON(b []byte) error {
	var v linuxSELinuxValue
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	l.val = v
	return nil
}

func (l LinuxSELinuxContext) User() LinuxSELinuxUser {
	return l.val.User
}

func (l LinuxSELinuxContext) Role() LinuxSELinuxRole {
	return l.val.Role
}

func (l LinuxSELinuxContext) Type() LinuxSELinuxType {
	return l.val.Type
}

func (l LinuxSELinuxContext) Level() LinuxSELinuxLevel {
	return l.val.Level
}

func (l LinuxSELinuxContext) multipleAllowed() bool {
	return false
}

func (l LinuxSELinuxContext) Conflicts() []ACIdentifier {
	return nil
}

func NewLinuxSELinuxContext(selinuxUser, selinuxRole, selinuxType, selinuxLevel string) (*LinuxSELinuxContext, error) {
	l := LinuxSELinuxContext{
		linuxSELinuxValue{
			LinuxSELinuxUser(selinuxUser),
			LinuxSELinuxRole(selinuxRole),
			LinuxSELinuxType(selinuxType),
			LinuxSELinuxLevel(selinuxLevel),
		},
	}
	if err := l.AssertValid(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l LinuxSELinuxContext) AsIsolator() (*Isolator, error) {
	b, err := json.Marshal(l.val)
	if err != nil {
		return nil, err
	}
	rm := json.RawMessage(b)
	return &Isolator{
		Name:     LinuxSELinuxContextName,
		ValueRaw: &rm,
		value:    &l,
	}, nil
}
