package types

import (
	"encoding/json"
	"errors"
)

const (
	LinuxCapabilitiesRetainSetName = "os/linux/capabilities-retain-set"
	LinuxCapabilitiesRevokeSetName = "os/linux/capabilities-revoke-set"
)

func init() {
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
