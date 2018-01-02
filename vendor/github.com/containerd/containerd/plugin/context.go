package plugin

import (
	"context"
	"path/filepath"

	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events/exchange"
	"github.com/containerd/containerd/log"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

// InitContext is used for plugin inititalization
type InitContext struct {
	Context context.Context
	Root    string
	State   string
	Config  interface{}
	Address string
	Events  *exchange.Exchange

	Meta *Meta // plugins can fill in metadata at init.

	plugins *Set
}

// NewContext returns a new plugin InitContext
func NewContext(ctx context.Context, r *Registration, plugins *Set, root, state string) *InitContext {
	return &InitContext{
		Context: log.WithModule(ctx, r.URI()),
		Root:    filepath.Join(root, r.URI()),
		State:   filepath.Join(state, r.URI()),
		Meta: &Meta{
			Exports: map[string]string{},
		},
		plugins: plugins,
	}
}

// Get returns the first plugin by its type
func (i *InitContext) Get(t Type) (interface{}, error) {
	return i.plugins.Get(t)
}

// Meta contains information gathered from the registration and initialization
// process.
type Meta struct {
	Platforms    []ocispec.Platform // platforms supported by plugin
	Exports      map[string]string  // values exported by plugin
	Capabilities []string           // feature switches for plugin
}

// Plugin represents an initialized plugin, used with an init context.
type Plugin struct {
	Registration *Registration // registration, as initialized
	Config       interface{}   // config, as initialized
	Meta         *Meta

	instance interface{}
	err      error // will be set if there was an error initializing the plugin
}

// Err returns the errors during initialization.
// returns nil if not error was encountered
func (p *Plugin) Err() error {
	return p.err
}

// Instance returns the instance and any initialization error of the plugin
func (p *Plugin) Instance() (interface{}, error) {
	return p.instance, p.err
}

// Set defines a plugin collection, used with InitContext.
//
// This maintains ordering and unique indexing over the set.
//
// After iteratively instantiating plugins, this set should represent, the
// ordered, initialization set of plugins for a containerd instance.
type Set struct {
	ordered     []*Plugin // order of initialization
	byTypeAndID map[Type]map[string]*Plugin
}

// NewPluginSet returns an initialized plugin set
func NewPluginSet() *Set {
	return &Set{
		byTypeAndID: make(map[Type]map[string]*Plugin),
	}
}

// Add a plugin to the set
func (ps *Set) Add(p *Plugin) error {
	if byID, typeok := ps.byTypeAndID[p.Registration.Type]; !typeok {
		ps.byTypeAndID[p.Registration.Type] = map[string]*Plugin{
			p.Registration.ID: p,
		}
	} else if _, idok := byID[p.Registration.ID]; !idok {
		byID[p.Registration.ID] = p
	} else {
		return errors.Wrapf(errdefs.ErrAlreadyExists, "plugin %v already initialized", p.Registration.URI())
	}

	ps.ordered = append(ps.ordered, p)
	return nil
}

// Get returns the first plugin by its type
func (ps *Set) Get(t Type) (interface{}, error) {
	for _, v := range ps.byTypeAndID[t] {
		return v.Instance()
	}
	return nil, errors.Wrapf(errdefs.ErrNotFound, "no plugins registered for %s", t)
}

// GetAll plugins in the set
func (i *InitContext) GetAll() []*Plugin {
	return i.plugins.ordered
}

// GetByType returns all plugins with the specific type.
func (i *InitContext) GetByType(t Type) (map[string]*Plugin, error) {
	p, ok := i.plugins.byTypeAndID[t]
	if !ok {
		return nil, errors.Wrapf(errdefs.ErrNotFound, "no plugins registered for %s", t)
	}

	return p, nil
}
