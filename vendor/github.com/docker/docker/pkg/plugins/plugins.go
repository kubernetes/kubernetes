package plugins

import (
	"errors"
	"sync"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/tlsconfig"
)

var (
	ErrNotImplements = errors.New("Plugin does not implement the requested driver")
)

type plugins struct {
	sync.Mutex
	plugins map[string]*Plugin
}

var (
	storage          = plugins{plugins: make(map[string]*Plugin)}
	extpointHandlers = make(map[string]func(string, *Client))
)

type Manifest struct {
	Implements []string
}

type Plugin struct {
	Name      string `json:"-"`
	Addr      string
	TLSConfig tlsconfig.Options
	Client    *Client   `json:"-"`
	Manifest  *Manifest `json:"-"`
}

func newLocalPlugin(name, addr string) *Plugin {
	return &Plugin{
		Name:      name,
		Addr:      addr,
		TLSConfig: tlsconfig.Options{InsecureSkipVerify: true},
	}
}

func (p *Plugin) activate() error {
	c, err := NewClient(p.Addr, p.TLSConfig)
	if err != nil {
		return err
	}
	p.Client = c

	m := new(Manifest)
	if err = p.Client.Call("Plugin.Activate", nil, m); err != nil {
		return err
	}

	logrus.Debugf("%s's manifest: %v", p.Name, m)
	p.Manifest = m

	for _, iface := range m.Implements {
		handler, handled := extpointHandlers[iface]
		if !handled {
			continue
		}
		handler(p.Name, p.Client)
	}
	return nil
}

func load(name string) (*Plugin, error) {
	registry := newLocalRegistry()
	pl, err := registry.Plugin(name)
	if err != nil {
		return nil, err
	}
	if err := pl.activate(); err != nil {
		return nil, err
	}
	return pl, nil
}

func get(name string) (*Plugin, error) {
	storage.Lock()
	defer storage.Unlock()
	pl, ok := storage.plugins[name]
	if ok {
		return pl, nil
	}
	pl, err := load(name)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("Plugin: %v", pl)
	storage.plugins[name] = pl
	return pl, nil
}

func Get(name, imp string) (*Plugin, error) {
	pl, err := get(name)
	if err != nil {
		return nil, err
	}
	for _, driver := range pl.Manifest.Implements {
		logrus.Debugf("%s implements: %s", name, driver)
		if driver == imp {
			return pl, nil
		}
	}
	return nil, ErrNotImplements
}

func Handle(iface string, fn func(string, *Client)) {
	extpointHandlers[iface] = fn
}
