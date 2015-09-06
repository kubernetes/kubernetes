/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remote

import (
	"errors"
	"github.com/docker/docker/pkg/tlsconfig"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/sock"
	"sync"
)

const (
	// NetworkController methods
	ActivateMethod           = "Activate"
	GetNetworkMethod         = "GetNetwork"
	CreateNetworkMethod      = "CreateNetwork"
	UpdateNetworkMethod      = "UpdateNetwork"
	CheckTenantIDMethod      = "CheckTenantID"
	DeleteNetworkMethod      = "DeleteNetwork"
	GetLoadBalancerMethod    = "GetLoadBalancer"
	CreateLoadBalancerMethod = "CreateLoadBalancer"
	UpdateLoadBalancerMethod = "UpdateLoadBalancer"
	DeleteLoadBalancerMethod = "DeleteLoadBalancer"

	// Kubelet methods
	SetupPodMethod    = "SetupPod"
	TeardownPodMethod = "TeardownPod"
	PodStatudMethod   = "PodStatus"
)

var (
	plugins = Plugins{plugins: make(map[string]*Plugin)}
)

type Plugins struct {
	sync.Mutex
	plugins map[string]*Plugin
}

// Plugin is the definition of a docker plugin.
type Plugin struct {
	// Name of the plugin
	Name string `json:"-"`
	// Address of the plugin
	Addr string
	// TLS configuration of the plugin
	TLSConfig tlsconfig.Options
	// Client attached to the plugin
	Client *sock.Client `json:"-"`

	activatErr   error
	activateOnce sync.Once
}

func newLocalPlugin(name, addr string) *Plugin {
	return &Plugin{
		Name:      name,
		Addr:      addr,
		TLSConfig: tlsconfig.Options{InsecureSkipVerify: true},
	}
}

func (p *Plugin) activate() error {
	p.activateOnce.Do(func() {
		p.activatErr = p.activateWithLock()
	})
	return p.activatErr
}

func (p *Plugin) activateWithLock() error {
	c, err := sock.NewClient(p.Addr, p.TLSConfig)
	if err != nil {
		return err
	}
	p.Client = c

	m := ActivateResponse{}
	if err = p.Client.Call(ActivateMethod, nil, &m); err != nil {
		glog.Warningf("Active network plugin %s failed: %v", p.Name, err)
		return err
	}

	if !m.Result {
		glog.Warningf("Active network plugin %s failed: %v", p.Name, m.GetError())
		return errors.New(m.GetError())
	}

	glog.V(4).Infof("%s's status: %v", p.Name, m)

	return nil
}

func load(name string) (*Plugin, error) {
	plugins.Lock()
	registry := newLocalRegistry()
	pl, err := registry.Plugin(name)
	if err == nil {
		plugins.plugins[name] = pl
	}
	plugins.Unlock()

	if err != nil {
		return nil, err
	}

	err = pl.activate()

	if err != nil {
		plugins.Lock()
		delete(plugins.plugins, name)
		plugins.Unlock()
	}

	return pl, err
}

func GetPlugin(name string) (*Plugin, error) {
	plugins.Lock()
	pl, ok := plugins.plugins[name]
	plugins.Unlock()
	if ok {
		return pl, pl.activate()
	}
	return load(name)
}
