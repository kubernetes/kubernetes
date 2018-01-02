package v2

import (
	"fmt"
	"strings"
	"sync"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/plugins"
	"github.com/opencontainers/go-digest"
)

// Plugin represents an individual plugin.
type Plugin struct {
	mu              sync.RWMutex
	PluginObj       types.Plugin `json:"plugin"` // todo: embed struct
	pClient         *plugins.Client
	refCount        int
	PropagatedMount string // TODO: make private
	Rootfs          string // TODO: make private

	Config   digest.Digest
	Blobsums []digest.Digest

	SwarmServiceID string
}

const defaultPluginRuntimeDestination = "/run/docker/plugins"

// ErrInadequateCapability indicates that the plugin did not have the requested capability.
type ErrInadequateCapability struct {
	cap string
}

func (e ErrInadequateCapability) Error() string {
	return fmt.Sprintf("plugin does not provide %q capability", e.cap)
}

// BasePath returns the path to which all paths returned by the plugin are relative to.
// For Plugin objects this returns the host path of the plugin container's rootfs.
func (p *Plugin) BasePath() string {
	return p.Rootfs
}

// Client returns the plugin client.
func (p *Plugin) Client() *plugins.Client {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.pClient
}

// SetPClient set the plugin client.
func (p *Plugin) SetPClient(client *plugins.Client) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.pClient = client
}

// IsV1 returns true for V1 plugins and false otherwise.
func (p *Plugin) IsV1() bool {
	return false
}

// Name returns the plugin name.
func (p *Plugin) Name() string {
	return p.PluginObj.Name
}

// FilterByCap query the plugin for a given capability.
func (p *Plugin) FilterByCap(capability string) (*Plugin, error) {
	capability = strings.ToLower(capability)
	for _, typ := range p.PluginObj.Config.Interface.Types {
		if typ.Capability == capability && typ.Prefix == "docker" {
			return p, nil
		}
	}
	return nil, ErrInadequateCapability{capability}
}

// InitEmptySettings initializes empty settings for a plugin.
func (p *Plugin) InitEmptySettings() {
	p.PluginObj.Settings.Mounts = make([]types.PluginMount, len(p.PluginObj.Config.Mounts))
	copy(p.PluginObj.Settings.Mounts, p.PluginObj.Config.Mounts)
	p.PluginObj.Settings.Devices = make([]types.PluginDevice, len(p.PluginObj.Config.Linux.Devices))
	copy(p.PluginObj.Settings.Devices, p.PluginObj.Config.Linux.Devices)
	p.PluginObj.Settings.Env = make([]string, 0, len(p.PluginObj.Config.Env))
	for _, env := range p.PluginObj.Config.Env {
		if env.Value != nil {
			p.PluginObj.Settings.Env = append(p.PluginObj.Settings.Env, fmt.Sprintf("%s=%s", env.Name, *env.Value))
		}
	}
	p.PluginObj.Settings.Args = make([]string, len(p.PluginObj.Config.Args.Value))
	copy(p.PluginObj.Settings.Args, p.PluginObj.Config.Args.Value)
}

// Set is used to pass arguments to the plugin.
func (p *Plugin) Set(args []string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.PluginObj.Enabled {
		return fmt.Errorf("cannot set on an active plugin, disable plugin before setting")
	}

	sets, err := newSettables(args)
	if err != nil {
		return err
	}

	// TODO(vieux): lots of code duplication here, needs to be refactored.

next:
	for _, s := range sets {
		// range over all the envs in the config
		for _, env := range p.PluginObj.Config.Env {
			// found the env in the config
			if env.Name == s.name {
				// is it settable ?
				if ok, err := s.isSettable(allowedSettableFieldsEnv, env.Settable); err != nil {
					return err
				} else if !ok {
					return fmt.Errorf("%q is not settable", s.prettyName())
				}
				// is it, so lets update the settings in memory
				updateSettingsEnv(&p.PluginObj.Settings.Env, &s)
				continue next
			}
		}

		// range over all the mounts in the config
		for _, mount := range p.PluginObj.Config.Mounts {
			// found the mount in the config
			if mount.Name == s.name {
				// is it settable ?
				if ok, err := s.isSettable(allowedSettableFieldsMounts, mount.Settable); err != nil {
					return err
				} else if !ok {
					return fmt.Errorf("%q is not settable", s.prettyName())
				}

				// it is, so lets update the settings in memory
				*mount.Source = s.value
				continue next
			}
		}

		// range over all the devices in the config
		for _, device := range p.PluginObj.Config.Linux.Devices {
			// found the device in the config
			if device.Name == s.name {
				// is it settable ?
				if ok, err := s.isSettable(allowedSettableFieldsDevices, device.Settable); err != nil {
					return err
				} else if !ok {
					return fmt.Errorf("%q is not settable", s.prettyName())
				}

				// it is, so lets update the settings in memory
				*device.Path = s.value
				continue next
			}
		}

		// found the name in the config
		if p.PluginObj.Config.Args.Name == s.name {
			// is it settable ?
			if ok, err := s.isSettable(allowedSettableFieldsArgs, p.PluginObj.Config.Args.Settable); err != nil {
				return err
			} else if !ok {
				return fmt.Errorf("%q is not settable", s.prettyName())
			}

			// it is, so lets update the settings in memory
			p.PluginObj.Settings.Args = strings.Split(s.value, " ")
			continue next
		}

		return fmt.Errorf("setting %q not found in the plugin configuration", s.name)
	}

	return nil
}

// IsEnabled returns the active state of the plugin.
func (p *Plugin) IsEnabled() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.PluginObj.Enabled
}

// GetID returns the plugin's ID.
func (p *Plugin) GetID() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.PluginObj.ID
}

// GetSocket returns the plugin socket.
func (p *Plugin) GetSocket() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.PluginObj.Config.Interface.Socket
}

// GetTypes returns the interface types of a plugin.
func (p *Plugin) GetTypes() []types.PluginInterfaceType {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.PluginObj.Config.Interface.Types
}

// GetRefCount returns the reference count.
func (p *Plugin) GetRefCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.refCount
}

// AddRefCount adds to reference count.
func (p *Plugin) AddRefCount(count int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.refCount += count
}

// Acquire increments the plugin's reference count
// This should be followed up by `Release()` when the plugin is no longer in use.
func (p *Plugin) Acquire() {
	p.AddRefCount(plugingetter.Acquire)
}

// Release decrements the plugin's reference count
// This should only be called when the plugin is no longer in use, e.g. with
// via `Acquire()` or getter.Get("name", "type", plugingetter.Acquire)
func (p *Plugin) Release() {
	p.AddRefCount(plugingetter.Release)
}
