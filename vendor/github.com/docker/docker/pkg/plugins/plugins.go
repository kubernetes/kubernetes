// Package plugins provides structures and helper functions to manage Docker
// plugins.
//
// Docker discovers plugins by looking for them in the plugin directory whenever
// a user or container tries to use one by name. UNIX domain socket files must
// be located under /run/docker/plugins, whereas spec files can be located
// either under /etc/docker/plugins or /usr/lib/docker/plugins. This is handled
// by the Registry interface, which lets you list all plugins or get a plugin by
// its name if it exists.
//
// The plugins need to implement an HTTP server and bind this to the UNIX socket
// or the address specified in the spec files.
// A handshake is send at /Plugin.Activate, and plugins are expected to return
// a Manifest with a list of of Docker subsystems which this plugin implements.
//
// In order to use a plugins, you can use the ``Get`` with the name of the
// plugin and the subsystem it implements.
//
//	plugin, err := plugins.Get("example", "VolumeDriver")
//	if err != nil {
//		return fmt.Errorf("Error looking up volume plugin example: %v", err)
//	}
package plugins

import (
	"errors"
	"sync"
	"time"

	"github.com/docker/go-connections/tlsconfig"
	"github.com/sirupsen/logrus"
)

var (
	// ErrNotImplements is returned if the plugin does not implement the requested driver.
	ErrNotImplements = errors.New("Plugin does not implement the requested driver")
)

type plugins struct {
	sync.Mutex
	plugins map[string]*Plugin
}

type extpointHandlers struct {
	sync.RWMutex
	extpointHandlers map[string][]func(string, *Client)
}

var (
	storage  = plugins{plugins: make(map[string]*Plugin)}
	handlers = extpointHandlers{extpointHandlers: make(map[string][]func(string, *Client))}
)

// Manifest lists what a plugin implements.
type Manifest struct {
	// List of subsystem the plugin implements.
	Implements []string
}

// Plugin is the definition of a docker plugin.
type Plugin struct {
	// Name of the plugin
	name string
	// Address of the plugin
	Addr string
	// TLS configuration of the plugin
	TLSConfig *tlsconfig.Options
	// Client attached to the plugin
	client *Client
	// Manifest of the plugin (see above)
	Manifest *Manifest `json:"-"`

	// wait for activation to finish
	activateWait *sync.Cond
	// error produced by activation
	activateErr error
	// keeps track of callback handlers run against this plugin
	handlersRun bool
}

// Name returns the name of the plugin.
func (p *Plugin) Name() string {
	return p.name
}

// Client returns a ready-to-use plugin client that can be used to communicate with the plugin.
func (p *Plugin) Client() *Client {
	return p.client
}

// IsV1 returns true for V1 plugins and false otherwise.
func (p *Plugin) IsV1() bool {
	return true
}

// NewLocalPlugin creates a new local plugin.
func NewLocalPlugin(name, addr string) *Plugin {
	return &Plugin{
		name: name,
		Addr: addr,
		// TODO: change to nil
		TLSConfig:    &tlsconfig.Options{InsecureSkipVerify: true},
		activateWait: sync.NewCond(&sync.Mutex{}),
	}
}

func (p *Plugin) activate() error {
	p.activateWait.L.Lock()

	if p.activated() {
		p.runHandlers()
		p.activateWait.L.Unlock()
		return p.activateErr
	}

	p.activateErr = p.activateWithLock()

	p.runHandlers()
	p.activateWait.L.Unlock()
	p.activateWait.Broadcast()
	return p.activateErr
}

// runHandlers runs the registered handlers for the implemented plugin types
// This should only be run after activation, and while the activation lock is held.
func (p *Plugin) runHandlers() {
	if !p.activated() {
		return
	}

	handlers.RLock()
	if !p.handlersRun {
		for _, iface := range p.Manifest.Implements {
			hdlrs, handled := handlers.extpointHandlers[iface]
			if !handled {
				continue
			}
			for _, handler := range hdlrs {
				handler(p.name, p.client)
			}
		}
		p.handlersRun = true
	}
	handlers.RUnlock()

}

// activated returns if the plugin has already been activated.
// This should only be called with the activation lock held
func (p *Plugin) activated() bool {
	return p.Manifest != nil
}

func (p *Plugin) activateWithLock() error {
	c, err := NewClient(p.Addr, p.TLSConfig)
	if err != nil {
		return err
	}
	p.client = c

	m := new(Manifest)
	if err = p.client.Call("Plugin.Activate", nil, m); err != nil {
		return err
	}

	p.Manifest = m
	return nil
}

func (p *Plugin) waitActive() error {
	p.activateWait.L.Lock()
	for !p.activated() && p.activateErr == nil {
		p.activateWait.Wait()
	}
	p.activateWait.L.Unlock()
	return p.activateErr
}

func (p *Plugin) implements(kind string) bool {
	if p.Manifest == nil {
		return false
	}
	for _, driver := range p.Manifest.Implements {
		if driver == kind {
			return true
		}
	}
	return false
}

func load(name string) (*Plugin, error) {
	return loadWithRetry(name, true)
}

func loadWithRetry(name string, retry bool) (*Plugin, error) {
	registry := newLocalRegistry()
	start := time.Now()

	var retries int
	for {
		pl, err := registry.Plugin(name)
		if err != nil {
			if !retry {
				return nil, err
			}

			timeOff := backoff(retries)
			if abort(start, timeOff) {
				return nil, err
			}
			retries++
			logrus.Warnf("Unable to locate plugin: %s, retrying in %v", name, timeOff)
			time.Sleep(timeOff)
			continue
		}

		storage.Lock()
		if pl, exists := storage.plugins[name]; exists {
			storage.Unlock()
			return pl, pl.activate()
		}
		storage.plugins[name] = pl
		storage.Unlock()

		err = pl.activate()

		if err != nil {
			storage.Lock()
			delete(storage.plugins, name)
			storage.Unlock()
		}

		return pl, err
	}
}

func get(name string) (*Plugin, error) {
	storage.Lock()
	pl, ok := storage.plugins[name]
	storage.Unlock()
	if ok {
		return pl, pl.activate()
	}
	return load(name)
}

// Get returns the plugin given the specified name and requested implementation.
func Get(name, imp string) (*Plugin, error) {
	pl, err := get(name)
	if err != nil {
		return nil, err
	}
	if err := pl.waitActive(); err == nil && pl.implements(imp) {
		logrus.Debugf("%s implements: %s", name, imp)
		return pl, nil
	}
	return nil, ErrNotImplements
}

// Handle adds the specified function to the extpointHandlers.
func Handle(iface string, fn func(string, *Client)) {
	handlers.Lock()
	hdlrs, ok := handlers.extpointHandlers[iface]
	if !ok {
		hdlrs = []func(string, *Client){}
	}

	hdlrs = append(hdlrs, fn)
	handlers.extpointHandlers[iface] = hdlrs

	storage.Lock()
	for _, p := range storage.plugins {
		p.activateWait.L.Lock()
		if p.activated() && p.implements(iface) {
			p.handlersRun = false
		}
		p.activateWait.L.Unlock()
	}
	storage.Unlock()

	handlers.Unlock()
}

// GetAll returns all the plugins for the specified implementation
func GetAll(imp string) ([]*Plugin, error) {
	pluginNames, err := Scan()
	if err != nil {
		return nil, err
	}

	type plLoad struct {
		pl  *Plugin
		err error
	}

	chPl := make(chan *plLoad, len(pluginNames))
	var wg sync.WaitGroup
	for _, name := range pluginNames {
		storage.Lock()
		pl, ok := storage.plugins[name]
		storage.Unlock()
		if ok {
			chPl <- &plLoad{pl, nil}
			continue
		}

		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			pl, err := loadWithRetry(name, false)
			chPl <- &plLoad{pl, err}
		}(name)
	}

	wg.Wait()
	close(chPl)

	var out []*Plugin
	for pl := range chPl {
		if pl.err != nil {
			logrus.Error(pl.err)
			continue
		}
		if err := pl.pl.waitActive(); err == nil && pl.pl.implements(imp) {
			out = append(out, pl.pl)
		}
	}
	return out, nil
}
