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
	"os"
	"testing"

	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/kubernetes/test/e2e/framework"

	"k8s.io/klog/v2"
)

// e2eService manages e2e services in current process.
type e2eServices struct {
	rmDirs []string
	// statically linked e2e services
	etcdServer   *etcd3testing.EtcdTestServer
	etcdStorage  *storagebackend.Config
	apiServer    *APIServer
	nsController *NamespaceController
}

func newE2EServices() *e2eServices {
	return &e2eServices{}
}

// run starts all e2e services and wait for the termination signal. Once receives the
// termination signal, it will stop the e2e services gracefully.
func (es *e2eServices) run(t *testing.T) error {
	defer es.stop(t)
	if err := es.start(t); err != nil {
		return err
	}
	// Wait until receiving a termination signal.
	waitForTerminationSignal()
	return nil
}

// start starts the tests embedded services or returns an error.
func (es *e2eServices) start(t *testing.T) error {
	klog.Info("Starting e2e services...")
	err := es.startEtcd(t)
	if err != nil {
		return err
	}
	err = es.startAPIServer(es.etcdStorage)
	if err != nil {
		return err
	}
	err = es.startNamespaceController()
	if err != nil {
		return nil
	}
	klog.Info("E2E services started.")
	return nil
}

// stop stops the embedded e2e services.
func (es *e2eServices) stop(t *testing.T) {
	klog.Info("Stopping e2e services...")
	// TODO(random-liu): Use a loop to stop all services after introducing
	// service interface.
	klog.Info("Stopping namespace controller")
	if es.nsController != nil {
		if err := es.nsController.Stop(); err != nil {
			klog.Errorf("Failed to stop %q: %v", es.nsController.Name(), err)
		}
	}

	klog.Info("Stopping API server")
	if es.apiServer != nil {
		if err := es.apiServer.Stop(); err != nil {
			klog.Errorf("Failed to stop %q: %v", es.apiServer.Name(), err)
		}
	}

	klog.Info("Stopping etcd")
	if es.etcdServer != nil {
		es.etcdServer.Terminate(t)
	}

	for _, d := range es.rmDirs {
		klog.Infof("Deleting directory %v", d)
		err := os.RemoveAll(d)
		if err != nil {
			klog.Errorf("Failed to delete directory %s.\n%v", d, err)
		}
	}

	klog.Info("E2E services stopped.")
}

// startEtcd starts the embedded etcd instance or returns an error.
func (es *e2eServices) startEtcd(t *testing.T) error {
	klog.Info("Starting etcd")
	server, etcdStorage := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	es.etcdServer = server
	es.etcdStorage = etcdStorage
	return nil
}

// startAPIServer starts the embedded API server or returns an error.
func (es *e2eServices) startAPIServer(etcdStorage *storagebackend.Config) error {
	klog.Info("Starting API server")
	es.apiServer = NewAPIServer(*etcdStorage)
	return es.apiServer.Start()
}

// startNamespaceController starts the embedded namespace controller or returns an error.
func (es *e2eServices) startNamespaceController() error {
	klog.Info("Starting namespace controller")
	es.nsController = NewNamespaceController(framework.TestContext.Host)
	return es.nsController.Start()
}

// getServicesHealthCheckURLs returns the health check urls for the internal services.
func getServicesHealthCheckURLs() []string {
	return []string{
		getAPIServerHealthCheckURL(),
	}
}
