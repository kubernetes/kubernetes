//go:build windows

package cni

import (
	"errors"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcnshim/internal/regstate"
)

const (
	cniRoot = "cni"
	cniKey  = "cfg"
)

// PersistedNamespaceConfig is the registry version of the `NamespaceID` to UVM
// map.
type PersistedNamespaceConfig struct {
	namespaceID string
	stored      bool

	ContainerID  string
	HostUniqueID guid.GUID
}

// NewPersistedNamespaceConfig creates an in-memory namespace config that can be
// persisted to the registry.
func NewPersistedNamespaceConfig(namespaceID, containerID string, containerHostUniqueID guid.GUID) *PersistedNamespaceConfig {
	return &PersistedNamespaceConfig{
		namespaceID:  namespaceID,
		ContainerID:  containerID,
		HostUniqueID: containerHostUniqueID,
	}
}

// LoadPersistedNamespaceConfig loads a persisted config from the registry that matches
// `namespaceID`. If not found returns `regstate.NotFoundError`
func LoadPersistedNamespaceConfig(namespaceID string) (*PersistedNamespaceConfig, error) {
	sk, err := regstate.Open(cniRoot, false)
	if err != nil {
		return nil, err
	}
	defer sk.Close()

	pnc := PersistedNamespaceConfig{
		namespaceID: namespaceID,
		stored:      true,
	}
	if err := sk.Get(namespaceID, cniKey, &pnc); err != nil {
		return nil, err
	}
	return &pnc, nil
}

// Store stores or updates the in-memory config to its registry state. If the
// store failes returns the store error.
func (pnc *PersistedNamespaceConfig) Store() error {
	if pnc.namespaceID == "" {
		return errors.New("invalid namespaceID ''")
	}
	if pnc.ContainerID == "" {
		return errors.New("invalid containerID ''")
	}
	empty := guid.GUID{}
	if pnc.HostUniqueID == empty {
		return errors.New("invalid containerHostUniqueID 'empy'")
	}
	sk, err := regstate.Open(cniRoot, false)
	if err != nil {
		return err
	}
	defer sk.Close()

	if pnc.stored {
		if err := sk.Set(pnc.namespaceID, cniKey, pnc); err != nil {
			return err
		}
	} else {
		if err := sk.Create(pnc.namespaceID, cniKey, pnc); err != nil {
			return err
		}
	}
	pnc.stored = true
	return nil
}

// Remove removes any persisted state associated with this config. If the config
// is not found in the registry `Remove` returns no error.
func (pnc *PersistedNamespaceConfig) Remove() error {
	if pnc.stored {
		sk, err := regstate.Open(cniRoot, false)
		if err != nil {
			if regstate.IsNotFoundError(err) {
				pnc.stored = false
				return nil
			}
			return err
		}
		defer sk.Close()

		if err := sk.Remove(pnc.namespaceID); err != nil {
			if regstate.IsNotFoundError(err) {
				pnc.stored = false
				return nil
			}
			return err
		}
	}
	pnc.stored = false
	return nil
}
