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
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/dns/fed"

	"time"

	"github.com/golang/glog"
)

// Sync manages synchronization of the config map.
type Sync interface {
	// Once does a single blocking synchronization of the config map. If
	// the ConfigMap fails to validate, this method will return nil,
	// err.
	Once() (*Config, error)

	// Start a periodic synchronization of the configuration map. When a
	// successful configuration map update is detected, the
	// configuration will be sent to the channel.
	Periodic() <-chan *Config
}

// NewSync for ConfigMap from namespace `ns` and `name`.
func NewSync(client clientset.Interface, ns string, name string, interval time.Duration) Sync {
	return &kubeSync{
		interval: interval,
		ns:       ns,
		name:     name,
		client:   client,
	}
}

// kubeSync implements Sync for the Kubernetes API.
type kubeSync struct {
	interval time.Duration

	ns            string
	name          string
	latestVersion string

	client clientset.Interface
}

func (sync *kubeSync) Once() (*Config, error) {
	config, _, err := sync.doSync()
	return config, err
}

func (sync *kubeSync) Periodic() <-chan *Config {
	syncChan := make(chan *Config)

	go func() {
		for {
			glog.V(4).Infof("Periodic sync waiting for %v", sync.interval)
			<-time.After(sync.interval)

			config, updated, err := sync.doSync()

			if updated && err != nil {
				syncChan <- config
			}
		}
	}()

	return syncChan
}

func (sync *kubeSync) doSync() (config *Config, changed bool, err error) {
	glog.V(5).Infof("doSync %+v", sync)

	var cm *api.ConfigMap
	cm, err = sync.client.Core().ConfigMaps(sync.ns).Get(sync.name)

	if err != nil {
		glog.Errorf("Error getting ConfigMap %v:%v err: %v",
			sync.ns, sync.name, err)
		return
	}

	glog.V(4).Infof("Got ConfigMap %+v", cm)

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
		return
	}

	if err = config.Validate(); err != nil {
		glog.Errorf("Invalid configuration: %v (value was %+v)", err, config)
		config = nil
		return
	}

	return
}

func (sync *kubeSync) updateFederations(cm *api.ConfigMap, config *Config) (err error) {
	if flagValue, ok := cm.Data["federations"]; ok {
		if err = fed.ParseFederationsFlag(flagValue, &config.Federations); err != nil {
			glog.Errorf("Invalid federations value: %v (value was %q)",
				err, cm.Data["federations"])
			return
		}
	} else {
		glog.V(4).Infof("No federations present")
	}

	return
}
