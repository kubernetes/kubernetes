/*
Copyright 2014 Google Inc. All rights reserved.

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

// Reads the configuration from the file. Example file for two services [nodejs & mysql]
//{"Services": [
//   {
//      "Name":"nodejs",
//      "Port":10000,
//      "Endpoints":["10.240.180.168:8000", "10.240.254.199:8000", "10.240.62.150:8000"]
//   },
//   {
//      "Name":"mysql",
//      "Port":10001,
//      "Endpoints":["10.240.180.168:9000", "10.240.254.199:9000", "10.240.62.150:9000"]
//   }
//]
//}

package config

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	fsnotify "gopkg.in/fsnotify.v0"
)

// serviceConfig is a deserialized form of the config file format which ConfigSourceFile accepts.
type serviceConfig struct {
	Services []struct {
		Name      string   `json: "name"`
		Port      int      `json: "port"`
		Endpoints []string `json: "endpoints"`
	} `json: "service"`
}

// ConfigSourceFile periodically reads service configurations in JSON from a file, and sends the services and endpoints defined in the file to the specified channels.
type ConfigSourceFile struct {
	serviceChannel   chan ServiceUpdate
	endpointsChannel chan EndpointsUpdate
	filename         string
}

// NewConfigSourceFile creates a new ConfigSourceFile and let it immediately runs the created ConfigSourceFile in a goroutine.
func NewConfigSourceFile(filename string, serviceChannel chan ServiceUpdate, endpointsChannel chan EndpointsUpdate) ConfigSourceFile {
	config := ConfigSourceFile{
		filename:         filepath.Clean(filename),
		serviceChannel:   serviceChannel,
		endpointsChannel: endpointsChannel,
	}
	go config.Run()
	return config
}

// Run begins watching the config file.
func (s ConfigSourceFile) Run() {

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		glog.Errorf("Error initializing filesystem watcher for %s: %v", s.filename, err)
		return
	}

	defer watcher.Close()

	if _, err := os.Stat(s.filename); err == nil {
		s.updateConfig()
	}

	go util.Forever(func() { fileWatcher(&s, watcher) }, 1)

	glog.Infof("Watching %s configuration file", s.filename)
	if err := watcher.Add(filepath.Dir(s.filename)); err != nil {
		glog.Fatalf("Error watching the %s file: %v", s.filename, err)
	}

	select {}
}

func fileWatcher(s *ConfigSourceFile, w *fsnotify.Watcher) {
	select {
	case ev := <-w.Events:
		if ev.Name != s.filename {
			return
		}
		switch ev.Op {
		case fsnotify.Write:
			s.updateConfig()
		case fsnotify.Remove:
			s.unloadConfig()
		}
	case err := <-w.Errors:
		glog.Infof("Config file watcher for %s failed: %v", s.filename, err)
	}
}

func (s ConfigSourceFile) unloadConfig() {
	glog.Infof("Unloading configuration for %s", s.filename)
	s.serviceChannel <- ServiceUpdate{Op: SET}
	s.endpointsChannel <- EndpointsUpdate{Op: SET}
}

func (s ConfigSourceFile) updateConfig() {
	var (
		lastServices  []api.Service
		lastEndpoints []api.Endpoints
	)
	data, err := ioutil.ReadFile(s.filename)
	if err != nil {
		glog.Errorf("Unable to read the configuration file %s: %v", s.filename, err)
		return
	}
	config := &serviceConfig{}
	if err = json.Unmarshal(data, config); err != nil {
		glog.Errorf("Couldn't unmarshal configuration from file : %s %v", data, err)
		return
	}
	// Ok, we have a valid configuration, send to channel for
	// rejiggering.
	glog.Infof("Updating configuration for %s", s.filename)
	newServices := make([]api.Service, len(config.Services))
	newEndpoints := make([]api.Endpoints, len(config.Services))
	for i, service := range config.Services {
		newServices[i] = api.Service{JSONBase: api.JSONBase{ID: service.Name}, Port: service.Port}
		newEndpoints[i] = api.Endpoints{JSONBase: api.JSONBase{ID: service.Name}, Endpoints: service.Endpoints}
	}
	if !reflect.DeepEqual(lastServices, newServices) {
		serviceUpdate := ServiceUpdate{Op: SET, Services: newServices}
		s.serviceChannel <- serviceUpdate
		lastServices = newServices
	}
	if !reflect.DeepEqual(lastEndpoints, newEndpoints) {
		endpointsUpdate := EndpointsUpdate{Op: SET, Endpoints: newEndpoints}
		s.endpointsChannel <- endpointsUpdate
		lastEndpoints = newEndpoints
	}
}
