/*
Copyright 2016 The Kubernetes Authors.

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

package config

import (
	"k8s.io/client-go/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	fed "k8s.io/kubernetes/pkg/dns/federation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"time"

	"github.com/golang/glog"
)

// Sync manages synchronization of the config map.
type Sync interface {
	// Once does a blocking synchronization of the config map. If the
	// ConfigMap fails to validate, this method will return nil, err.
	Once() (*Config, error)

	// Start a periodic synchronization of the configuration map. When a
	// successful configuration map update is detected, the
	// configuration will be sent to the channel.
	//
	// It is an error to call this more than once.
	Periodic() <-chan *Config
}

// NewSync for ConfigMap from namespace `ns` and `name`.
func NewSync(client clientset.Interface, ns string, name string) Sync {
	sync := &kubeSync{
		ns:      ns,
		name:    name,
		client:  client,
		channel: make(chan *Config),
	}

	listWatch := &cache.ListWatch{
		ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fields.Set{"metadata.name": name}.AsSelector().String()
			return client.Core().ConfigMaps(ns).List(options)
		},
		WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fields.Set{"metadata.name": name}.AsSelector().String()
			return client.Core().ConfigMaps(ns).Watch(options)
		},
	}

	store, controller := cache.NewInformer(
		listWatch,
		&v1.ConfigMap{},
		time.Duration(0),
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sync.onAdd,
			DeleteFunc: sync.onDelete,
			UpdateFunc: sync.onUpdate,
		})

	sync.store = store
	sync.controller = controller

	return sync
}

// kubeSync implements Sync for the Kubernetes API.
type kubeSync struct {
	ns   string
	name string

	client     clientset.Interface
	store      cache.Store
	controller *cache.Controller

	channel chan *Config

	latestVersion string
}

var _ Sync = (*kubeSync)(nil)

func (sync *kubeSync) Once() (*Config, error) {
	cm, err := sync.client.Core().ConfigMaps(sync.ns).Get(sync.name)

	if err != nil {
		glog.Errorf("Error getting ConfigMap %v:%v err: %v",
			sync.ns, sync.name, err)
		return nil, err
	}

	config, _, err := sync.processUpdate(cm)
	return config, err
}

func (sync *kubeSync) Periodic() <-chan *Config {
	go sync.controller.Run(wait.NeverStop)
	return sync.channel
}

func (sync *kubeSync) toConfigMap(obj interface{}) *v1.ConfigMap {
	cm, ok := obj.(*v1.ConfigMap)
	if !ok {
		glog.Fatalf("Expected ConfigMap, got %T", obj)
	}
	return cm
}

func (sync *kubeSync) onAdd(obj interface{}) {
	cm := sync.toConfigMap(obj)

	glog.V(2).Infof("ConfigMap %s:%s was created", sync.ns, sync.name)

	config, updated, err := sync.processUpdate(cm)
	if updated && err == nil {
		sync.channel <- config
	}
}

func (sync *kubeSync) onDelete(_ interface{}) {
	glog.V(2).Infof("ConfigMap %s:%s was deleted, reverting to default configuration",
		sync.ns, sync.name)

	sync.latestVersion = ""
	sync.channel <- NewDefaultConfig()
}

func (sync *kubeSync) onUpdate(_, obj interface{}) {
	cm := sync.toConfigMap(obj)

	glog.V(2).Infof("ConfigMap %s:%s was updated", sync.ns, sync.name)

	config, changed, err := sync.processUpdate(cm)

	if changed && err == nil {
		sync.channel <- config
	}
}

func (sync *kubeSync) processUpdate(cm *v1.ConfigMap) (config *Config, changed bool, err error) {
	glog.V(4).Infof("processUpdate ConfigMap %+v", *cm)

	if cm.ObjectMeta.ResourceVersion != sync.latestVersion {
		glog.V(3).Infof("Updating config to version %v (was %v)",
			cm.ObjectMeta.ResourceVersion, sync.latestVersion)
		changed = true
		sync.latestVersion = cm.ObjectMeta.ResourceVersion
	} else {
		glog.V(4).Infof("Config was unchanged (version %v)", sync.latestVersion)
		return
	}

	config = &Config{}

	if err = sync.updateFederations(cm, config); err != nil {
		glog.Errorf("Invalid configuration, ignoring update")
		return
	}

	if err = config.Validate(); err != nil {
		glog.Errorf("Invalid onfiguration: %v (value was %+v), ignoring update",
			err, config)
		config = nil
		return
	}

	return
}

func (sync *kubeSync) updateFederations(cm *v1.ConfigMap, config *Config) (err error) {
	if flagValue, ok := cm.Data["federations"]; ok {
		config.Federations = make(map[string]string)
		if err = fed.ParseFederationsFlag(flagValue, config.Federations); err != nil {
			glog.Errorf("Invalid federations value: %v (value was %q)",
				err, cm.Data["federations"])
			return
		}
		glog.V(2).Infof("Updated federations to %v", config.Federations)
	} else {
		glog.V(2).Infof("No federations present")
	}

	return
}
