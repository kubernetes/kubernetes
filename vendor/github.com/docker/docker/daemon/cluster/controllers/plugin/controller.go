package plugin

import (
	"io"
	"io/ioutil"
	"net/http"

	"github.com/docker/distribution/reference"
	enginetypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm/runtime"
	"github.com/docker/docker/plugin"
	"github.com/docker/docker/plugin/v2"
	"github.com/docker/swarmkit/api"
	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// Controller is the controller for the plugin backend.
// Plugins are managed as a singleton object with a desired state (different from containers).
// With the the plugin controller instead of having a strict create->start->stop->remove
// task lifecycle like containers, we manage the desired state of the plugin and let
// the plugin manager do what it already does and monitor the plugin.
// We'll also end up with many tasks all pointing to the same plugin ID.
//
// TODO(@cpuguy83): registry auth is intentionally not supported until we work out
// the right way to pass registry crednetials via secrets.
type Controller struct {
	backend Backend
	spec    runtime.PluginSpec
	logger  *logrus.Entry

	pluginID  string
	serviceID string
	taskID    string

	// hook used to signal tests that `Wait()` is actually ready and waiting
	signalWaitReady func()
}

// Backend is the interface for interacting with the plugin manager
// Controller actions are passed to the configured backend to do the real work.
type Backend interface {
	Disable(name string, config *enginetypes.PluginDisableConfig) error
	Enable(name string, config *enginetypes.PluginEnableConfig) error
	Remove(name string, config *enginetypes.PluginRmConfig) error
	Pull(ctx context.Context, ref reference.Named, name string, metaHeaders http.Header, authConfig *enginetypes.AuthConfig, privileges enginetypes.PluginPrivileges, outStream io.Writer, opts ...plugin.CreateOpt) error
	Upgrade(ctx context.Context, ref reference.Named, name string, metaHeaders http.Header, authConfig *enginetypes.AuthConfig, privileges enginetypes.PluginPrivileges, outStream io.Writer) error
	Get(name string) (*v2.Plugin, error)
	SubscribeEvents(buffer int, events ...plugin.Event) (eventCh <-chan interface{}, cancel func())
}

// NewController returns a new cluster plugin controller
func NewController(backend Backend, t *api.Task) (*Controller, error) {
	spec, err := readSpec(t)
	if err != nil {
		return nil, err
	}
	return &Controller{
		backend:   backend,
		spec:      spec,
		serviceID: t.ServiceID,
		logger: logrus.WithFields(logrus.Fields{
			"controller": "plugin",
			"task":       t.ID,
			"plugin":     spec.Name,
		})}, nil
}

func readSpec(t *api.Task) (runtime.PluginSpec, error) {
	var cfg runtime.PluginSpec

	generic := t.Spec.GetGeneric()
	if err := proto.Unmarshal(generic.Payload.Value, &cfg); err != nil {
		return cfg, errors.Wrap(err, "error reading plugin spec")
	}
	return cfg, nil
}

// Update is the update phase from swarmkit
func (p *Controller) Update(ctx context.Context, t *api.Task) error {
	p.logger.Debug("Update")
	return nil
}

// Prepare is the prepare phase from swarmkit
func (p *Controller) Prepare(ctx context.Context) (err error) {
	p.logger.Debug("Prepare")

	remote, err := reference.ParseNormalizedNamed(p.spec.Remote)
	if err != nil {
		return errors.Wrapf(err, "error parsing remote reference %q", p.spec.Remote)
	}

	if p.spec.Name == "" {
		p.spec.Name = remote.String()
	}

	var authConfig enginetypes.AuthConfig
	privs := convertPrivileges(p.spec.Privileges)

	pl, err := p.backend.Get(p.spec.Name)

	defer func() {
		if pl != nil && err == nil {
			pl.Acquire()
		}
	}()

	if err == nil && pl != nil {
		if pl.SwarmServiceID != p.serviceID {
			return errors.Errorf("plugin already exists: %s", p.spec.Name)
		}
		if pl.IsEnabled() {
			if err := p.backend.Disable(pl.GetID(), &enginetypes.PluginDisableConfig{ForceDisable: true}); err != nil {
				p.logger.WithError(err).Debug("could not disable plugin before running upgrade")
			}
		}
		p.pluginID = pl.GetID()
		return p.backend.Upgrade(ctx, remote, p.spec.Name, nil, &authConfig, privs, ioutil.Discard)
	}

	if err := p.backend.Pull(ctx, remote, p.spec.Name, nil, &authConfig, privs, ioutil.Discard, plugin.WithSwarmService(p.serviceID)); err != nil {
		return err
	}
	pl, err = p.backend.Get(p.spec.Name)
	if err != nil {
		return err
	}
	p.pluginID = pl.GetID()

	return nil
}

// Start is the start phase from swarmkit
func (p *Controller) Start(ctx context.Context) error {
	p.logger.Debug("Start")

	pl, err := p.backend.Get(p.pluginID)
	if err != nil {
		return err
	}

	if p.spec.Disabled {
		if pl.IsEnabled() {
			return p.backend.Disable(p.pluginID, &enginetypes.PluginDisableConfig{ForceDisable: false})
		}
		return nil
	}
	if !pl.IsEnabled() {
		return p.backend.Enable(p.pluginID, &enginetypes.PluginEnableConfig{Timeout: 30})
	}
	return nil
}

// Wait causes the task to wait until returned
func (p *Controller) Wait(ctx context.Context) error {
	p.logger.Debug("Wait")

	pl, err := p.backend.Get(p.pluginID)
	if err != nil {
		return err
	}

	events, cancel := p.backend.SubscribeEvents(1, plugin.EventDisable{Plugin: pl.PluginObj}, plugin.EventRemove{Plugin: pl.PluginObj}, plugin.EventEnable{Plugin: pl.PluginObj})
	defer cancel()

	if p.signalWaitReady != nil {
		p.signalWaitReady()
	}

	if !p.spec.Disabled != pl.IsEnabled() {
		return errors.New("mismatched plugin state")
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case e := <-events:
			p.logger.Debugf("got event %#T", e)

			switch e.(type) {
			case plugin.EventEnable:
				if p.spec.Disabled {
					return errors.New("plugin enabled")
				}
			case plugin.EventRemove:
				return errors.New("plugin removed")
			case plugin.EventDisable:
				if !p.spec.Disabled {
					return errors.New("plugin disabled")
				}
			}
		}
	}
}

func isNotFound(err error) bool {
	_, ok := errors.Cause(err).(plugin.ErrNotFound)
	return ok
}

// Shutdown is the shutdown phase from swarmkit
func (p *Controller) Shutdown(ctx context.Context) error {
	p.logger.Debug("Shutdown")
	return nil
}

// Terminate is the terminate phase from swarmkit
func (p *Controller) Terminate(ctx context.Context) error {
	p.logger.Debug("Terminate")
	return nil
}

// Remove is the remove phase from swarmkit
func (p *Controller) Remove(ctx context.Context) error {
	p.logger.Debug("Remove")

	pl, err := p.backend.Get(p.pluginID)
	if err != nil {
		if isNotFound(err) {
			return nil
		}
		return err
	}

	pl.Release()
	if pl.GetRefCount() > 0 {
		p.logger.Debug("skipping remove due to ref count")
		return nil
	}

	// This may error because we have exactly 1 plugin, but potentially multiple
	// tasks which are calling remove.
	err = p.backend.Remove(p.pluginID, &enginetypes.PluginRmConfig{ForceRemove: true})
	if isNotFound(err) {
		return nil
	}
	return err
}

// Close is the close phase from swarmkit
func (p *Controller) Close() error {
	p.logger.Debug("Close")
	return nil
}

func convertPrivileges(ls []*runtime.PluginPrivilege) enginetypes.PluginPrivileges {
	var out enginetypes.PluginPrivileges
	for _, p := range ls {
		pp := enginetypes.PluginPrivilege{
			Name:        p.Name,
			Description: p.Description,
			Value:       p.Value,
		}
		out = append(out, pp)
	}
	return out
}
