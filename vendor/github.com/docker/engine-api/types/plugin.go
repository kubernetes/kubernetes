// +build experimental

package types

import (
	"encoding/json"
	"fmt"
)

// PluginInstallOptions holds parameters to install a plugin.
type PluginInstallOptions struct {
	Disabled              bool
	AcceptAllPermissions  bool
	RegistryAuth          string // RegistryAuth is the base64 encoded credentials for the registry
	PrivilegeFunc         RequestPrivilegeFunc
	AcceptPermissionsFunc func(PluginPrivileges) (bool, error)
}

// PluginConfig represents the values of settings potentially modifiable by a user
type PluginConfig struct {
	Mounts  []PluginMount
	Env     []string
	Args    []string
	Devices []PluginDevice
}

// Plugin represents a Docker plugin for the remote API
type Plugin struct {
	ID       string `json:"Id,omitempty"`
	Name     string
	Tag      string
	Active   bool
	Config   PluginConfig
	Manifest PluginManifest
}

// PluginsListResponse contains the response for the remote API
type PluginsListResponse []*Plugin

const (
	authzDriver   = "AuthzDriver"
	graphDriver   = "GraphDriver"
	ipamDriver    = "IpamDriver"
	networkDriver = "NetworkDriver"
	volumeDriver  = "VolumeDriver"
)

// PluginInterfaceType represents a type that a plugin implements.
type PluginInterfaceType struct {
	Prefix     string // This is always "docker"
	Capability string // Capability should be validated against the above list.
	Version    string // Plugin API version. Depends on the capability
}

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

// PluginInterface describes the interface between Docker and plugin
type PluginInterface struct {
	Types  []PluginInterfaceType
	Socket string
}

// PluginSetting is to be embedded in other structs, if they are supposed to be
// modifiable by the user.
type PluginSetting struct {
	Name        string
	Description string
	Settable    []string
}

// PluginNetwork represents the network configuration for a plugin
type PluginNetwork struct {
	Type string
}

// PluginMount represents the mount configuration for a plugin
type PluginMount struct {
	PluginSetting
	Source      *string
	Destination string
	Type        string
	Options     []string
}

// PluginEnv represents an environment variable for a plugin
type PluginEnv struct {
	PluginSetting
	Value *string
}

// PluginArgs represents the command line arguments for a plugin
type PluginArgs struct {
	PluginSetting
	Value []string
}

// PluginDevice represents a device for a plugin
type PluginDevice struct {
	PluginSetting
	Path *string
}

// PluginUser represents the user for the plugin's process
type PluginUser struct {
	UID uint32 `json:"Uid,omitempty"`
	GID uint32 `json:"Gid,omitempty"`
}

// PluginManifest represents the manifest of a plugin
type PluginManifest struct {
	ManifestVersion string
	Description     string
	Documentation   string
	Interface       PluginInterface
	Entrypoint      []string
	Workdir         string
	User            PluginUser `json:",omitempty"`
	Network         PluginNetwork
	Capabilities    []string
	Mounts          []PluginMount
	Devices         []PluginDevice
	Env             []PluginEnv
	Args            PluginArgs
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
