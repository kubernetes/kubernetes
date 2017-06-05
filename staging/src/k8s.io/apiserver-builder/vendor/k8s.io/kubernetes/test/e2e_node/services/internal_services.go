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

package services

import (
	"io/ioutil"
	"os"

	"github.com/golang/glog"
)

// e2eService manages e2e services in current process.
type e2eServices struct {
	rmDirs []string
	// statically linked e2e services
	etcdServer   *EtcdServer
	apiServer    *APIServer
	nsController *NamespaceController
}

func newE2EServices() *e2eServices {
	return &e2eServices{}
}

// run starts all e2e services and wait for the termination signal. Once receives the
// termination signal, it will stop the e2e services gracefully.
func (es *e2eServices) run() error {
	defer es.stop()
	if err := es.start(); err != nil {
		return err
	}
	// Wait until receiving a termination signal.
	waitForTerminationSignal()
	return nil
}

// start starts the tests embedded services or returns an error.
func (es *e2eServices) start() error {
	glog.Info("Starting e2e services...")
	err := es.startEtcd()
	if err != nil {
		return err
	}
	err = es.startApiServer()
	if err != nil {
		return err
	}
	err = es.startNamespaceController()
	if err != nil {
		return nil
	}
	glog.Info("E2E services started.")
	return nil
}

// stop stops the embedded e2e services.
func (es *e2eServices) stop() {
	glog.Info("Stopping e2e services...")
	// TODO(random-liu): Use a loop to stop all services after introducing
	// service interface.
	glog.Info("Stopping namespace controller")
	if es.nsController != nil {
		if err := es.nsController.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.nsController.Name(), err)
		}
	}

	glog.Info("Stopping API server")
	if es.apiServer != nil {
		if err := es.apiServer.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.apiServer.Name(), err)
		}
	}

	glog.Info("Stopping etcd")
	if es.etcdServer != nil {
		if err := es.etcdServer.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.etcdServer.Name(), err)
		}
	}

	for _, d := range es.rmDirs {
		glog.Info("Deleting directory %v", d)
		err := os.RemoveAll(d)
		if err != nil {
			glog.Errorf("Failed to delete directory %s.\n%v", d, err)
		}
	}

	glog.Info("E2E services stopped.")
}

// startEtcd starts the embedded etcd instance or returns an error.
func (es *e2eServices) startEtcd() error {
	glog.Info("Starting etcd")
	// Create data directory in current working space.
	dataDir, err := ioutil.TempDir(".", "etcd")
	if err != nil {
		return err
	}
	// Mark the dataDir as directories to remove.
	es.rmDirs = append(es.rmDirs, dataDir)
	es.etcdServer = NewEtcd(dataDir)
	return es.etcdServer.Start()
}

// startApiServer starts the embedded API server or returns an error.
func (es *e2eServices) startApiServer() error {
	glog.Info("Starting API server")
	es.apiServer = NewAPIServer()
	return es.apiServer.Start()
}

// startNamespaceController starts the embedded namespace controller or returns an error.
func (es *e2eServices) startNamespaceController() error {
	glog.Info("Starting namespace controller")
	es.nsController = NewNamespaceController()
	return es.nsController.Start()
}

// getServicesHealthCheckURLs returns the health check urls for the internal services.
func getServicesHealthCheckURLs() []string {
	return []string{
		getEtcdHealthCheckURL(),
		getAPIServerHealthCheckURL(),
	}
}
