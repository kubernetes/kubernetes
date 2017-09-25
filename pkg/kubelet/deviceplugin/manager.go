/*
Copyright 2017 The Kubernetes Authors.

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

package deviceplugin

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

// ManagerImpl is the structure in charge of managing Device Plugins.
type ManagerImpl struct {
	socketname string
	socketdir  string

	endpoints map[string]*endpoint // Key is ResourceName
	mutex     sync.Mutex

	callback MonitorCallback

	server *grpc.Server
}

// NewManagerImpl creates a new manager on the socket `socketPath`.
// f is the callback that is called when a device becomes unhealthy.
// socketPath is present for testing purposes in production this is pluginapi.KubeletSocket
func NewManagerImpl(socketPath string, f MonitorCallback) (*ManagerImpl, error) {
	glog.V(2).Infof("Creating Device Plugin manager at %s", socketPath)

	if socketPath == "" || !filepath.IsAbs(socketPath) {
		return nil, fmt.Errorf(errBadSocket+" %v", socketPath)
	}

	dir, file := filepath.Split(socketPath)
	return &ManagerImpl{
		endpoints: make(map[string]*endpoint),

		socketname: file,
		socketdir:  dir,
		callback:   f,
	}, nil
}

func (m *ManagerImpl) removeContents(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		filePath := filepath.Join(dir, name)
		if filePath == m.CheckpointFile() {
			continue
		}
		stat, err := os.Stat(filePath)
		if err != nil {
			glog.Errorf("Failed to stat file %v: %v", filePath, err)
			continue
		}
		if stat.IsDir() {
			continue
		}
		err = os.RemoveAll(filePath)
		if err != nil {
			return err
		}
	}
	return nil
}

// CheckpointFile returns device plugin checkpoint file path.
func (m *ManagerImpl) CheckpointFile() string {
	return filepath.Join(m.socketdir, "kubelet_internal_checkpoint")
}

// Start starts the Device Plugin Manager
func (m *ManagerImpl) Start() error {
	glog.V(2).Infof("Starting Device Plugin manager")

	socketPath := filepath.Join(m.socketdir, m.socketname)
	os.MkdirAll(m.socketdir, 0755)

	// Removes all stale sockets in m.socketdir. Device plugins can monitor
	// this and use it as a signal to re-register with the new Kubelet.
	if err := m.removeContents(m.socketdir); err != nil {
		glog.Errorf("Fail to clean up stale contents under %s: %+v", m.socketdir, err)
	}

	if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
		glog.Errorf(errRemoveSocket+" %+v", err)
		return err
	}

	s, err := net.Listen("unix", socketPath)
	if err != nil {
		glog.Errorf(errListenSocket+" %+v", err)
		return err
	}

	m.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterRegistrationServer(m.server, m)
	go m.server.Serve(s)

	return nil
}

// Devices is the map of devices that are known by the Device
// Plugin manager with the kind of the devices as key
func (m *ManagerImpl) Devices() map[string][]*pluginapi.Device {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	devs := make(map[string][]*pluginapi.Device)
	for k, e := range m.endpoints {
		glog.V(3).Infof("Endpoint: %+v: %+v", k, e)
		devs[k] = e.getDevices()
	}

	return devs
}

// Allocate is the call that you can use to allocate a set of devices
// from the registered device plugins.
func (m *ManagerImpl) Allocate(resourceName string, devs []string) (*pluginapi.AllocateResponse, error) {

	if len(devs) == 0 {
		return nil, nil
	}

	glog.V(3).Infof("Recieved allocation request for devices %v for device plugin %s",
		devs, resourceName)
	m.mutex.Lock()
	e, ok := m.endpoints[resourceName]
	m.mutex.Unlock()
	if !ok {
		return nil, fmt.Errorf("Unknown Device Plugin %s", resourceName)
	}

	return e.allocate(devs)
}

// Register registers a device plugin.
func (m *ManagerImpl) Register(ctx context.Context,
	r *pluginapi.RegisterRequest) (*pluginapi.Empty, error) {
	glog.V(2).Infof("Got request for Device Plugin %s", r.ResourceName)
	if r.Version != pluginapi.Version {
		return &pluginapi.Empty{}, fmt.Errorf(errUnsuportedVersion)
	}

	if err := IsResourceNameValid(r.ResourceName); err != nil {
		return &pluginapi.Empty{}, err
	}

	// TODO: for now, always accepts newest device plugin. Later may consider to
	// add some policies here, e.g., verify whether an old device plugin with the
	// same resource name is still alive to determine whether we want to accept
	// the new registration.
	go m.addEndpoint(r)

	return &pluginapi.Empty{}, nil
}

// Stop is the function that can stop the gRPC server.
func (m *ManagerImpl) Stop() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	for _, e := range m.endpoints {
		e.stop()
	}

	m.server.Stop()
	return nil
}

func (m *ManagerImpl) addEndpoint(r *pluginapi.RegisterRequest) {
	socketPath := filepath.Join(m.socketdir, r.Endpoint)
	e, err := newEndpoint(socketPath, r.ResourceName, m.callback)
	if err != nil {
		glog.Errorf("Failed to dial device plugin with request %v: %v", r, err)
		return
	}
	stream, err := e.list()
	if err != nil {
		glog.Errorf("Failed to List devices for plugin %v: %v", r.ResourceName, err)
		return
	}

	// Associates the newly created endpoint with the corresponding resource name.
	// Stops existing endpoint if there is any.
	m.mutex.Lock()
	old, ok := m.endpoints[r.ResourceName]
	m.endpoints[r.ResourceName] = e
	m.mutex.Unlock()
	glog.V(2).Infof("Registered endpoint %v", e)
	if ok && old != nil {
		old.stop()
	}

	go func() {
		e.listAndWatch(stream)

		m.mutex.Lock()
		if old, ok := m.endpoints[r.ResourceName]; ok && old == e {
			glog.V(2).Infof("Delete resource for endpoint %v", e)
			delete(m.endpoints, r.ResourceName)
			// Issues callback to delete all of devices.
			e.callback(e.resourceName, []*pluginapi.Device{}, []*pluginapi.Device{}, e.getDevices())
		}
		glog.V(2).Infof("Unregistered endpoint %v", e)
		m.mutex.Unlock()
	}()
}
