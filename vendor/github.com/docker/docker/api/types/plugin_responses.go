package types

import (
	"encoding/json"
	"fmt"
)

// PluginsListResponse contains the response for the Engine API
type PluginsListResponse []*Plugin

const (
	authzDriver   = "AuthzDriver"
	graphDriver   = "GraphDriver"
	ipamDriver    = "IpamDriver"
	networkDriver = "NetworkDriver"
	volumeDriver  = "VolumeDriver"
)

// UnmarshalJSON implements json.Unmarshaler for PluginInterfaceType
func (t *PluginInterfaceType) UnmarshalJSON(p []byte) error {
	versionIndex := len(p)
	prefixIndex := 0
	if len(p) < 2 || p[0] != '"' || p[len(p)-1] != '"' {
		return fmt.Errorf("%q is not a plugin interface type", p)
	}
	p = p[1 : len(p)-1]
loop:
	for i, b := range p {
		switch b {
		case '.':
			prefixIndex = i
		case '/':
			versionIndex = i
			break loop
		}
	}
	t.Prefix = string(p[:prefixIndex])
	t.Capability = string(p[prefixIndex+1 : versionIndex])
	if versionIndex < len(p) {
		t.Version = string(p[versionIndex+1:])
	}
	return nil
}

// MarshalJSON implements json.Marshaler for PluginInterfaceType
func (t *PluginInterfaceType) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.String())
}

// String implements fmt.Stringer for PluginInterfaceType
func (t PluginInterfaceType) String() string {
	return fmt.Sprintf("%s.%s/%s", t.Prefix, t.Capability, t.Version)
}

// PluginPrivilege describes a permission the user has to accept
// upon installing a plugin.
type PluginPrivilege struct {
	Name        string
	Description string
	Value       []string
}

// PluginPrivileges is a list of PluginPrivilege
type PluginPrivileges []PluginPrivilege
