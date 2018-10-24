/*
Copyright 2018 The Kubernetes Authors.

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

package pluginwatcher

import (
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1alpha1"
	v1beta1 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta1"
	v1beta2 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta2"
)

// examplePlugin is a sample plugin to work with plugin watcher
type examplePlugin struct {
	grpcServer         *grpc.Server
	wg                 sync.WaitGroup
	registrationStatus chan registerapi.RegistrationStatus // for testing
	endpoint           string                              // for testing
	pluginName         string
	pluginType         string
	versions           []string
}

type pluginServiceV1Beta1 struct {
	server *examplePlugin
}

func (s *pluginServiceV1Beta1) GetExampleInfo(ctx context.Context, rqt *v1beta1.ExampleRequest) (*v1beta1.ExampleResponse, error) {
	glog.Infof("GetExampleInfo v1beta1field: %s", rqt.V1Beta1Field)
	return &v1beta1.ExampleResponse{}, nil
}

func (s *pluginServiceV1Beta1) RegisterService() {
	v1beta1.RegisterExampleServer(s.server.grpcServer, s)
}

type pluginServiceV1Beta2 struct {
	server *examplePlugin
}

func (s *pluginServiceV1Beta2) GetExampleInfo(ctx context.Context, rqt *v1beta2.ExampleRequest) (*v1beta2.ExampleResponse, error) {
	glog.Infof("GetExampleInfo v1beta2_field: %s", rqt.V1Beta2Field)
	return &v1beta2.ExampleResponse{}, nil
}

func (s *pluginServiceV1Beta2) RegisterService() {
	v1beta2.RegisterExampleServer(s.server.grpcServer, s)
}

// NewExamplePlugin returns an initialized examplePlugin instance
func NewExamplePlugin() *examplePlugin {
	return &examplePlugin{}
}

// NewTestExamplePlugin returns an initialized examplePlugin instance for testing
func NewTestExamplePlugin(pluginName string, pluginType string, endpoint string, advertisedVersions ...string) *examplePlugin {
	return &examplePlugin{
		pluginName:         pluginName,
		pluginType:         pluginType,
		endpoint:           endpoint,
		versions:           advertisedVersions,
		registrationStatus: make(chan registerapi.RegistrationStatus),
	}
}

// GetInfo is the RPC invoked by plugin watcher
func (e *examplePlugin) GetInfo(ctx context.Context, req *registerapi.InfoRequest) (*registerapi.PluginInfo, error) {
	return &registerapi.PluginInfo{
		Type:              e.pluginType,
		Name:              e.pluginName,
		Endpoint:          e.endpoint,
		SupportedVersions: e.versions,
	}, nil
}

func (e *examplePlugin) NotifyRegistrationStatus(ctx context.Context, status *registerapi.RegistrationStatus) (*registerapi.RegistrationStatusResponse, error) {
	glog.Errorf("Registration is: %v\n", status)

	if e.registrationStatus != nil {
		e.registrationStatus <- *status
	}

	return &registerapi.RegistrationStatusResponse{}, nil
}

// Serve starts a pluginwatcher server and one or more of the plugin services
func (e *examplePlugin) Serve(services ...string) error {
	glog.Infof("starting example server at: %s\n", e.endpoint)
	lis, err := net.Listen("unix", e.endpoint)
	if err != nil {
		return err
	}

	glog.Infof("example server started at: %s\n", e.endpoint)
	e.grpcServer = grpc.NewServer()

	// Registers kubelet plugin watcher api.
	registerapi.RegisterRegistrationServer(e.grpcServer, e)

	for _, service := range services {
		switch service {
		case "v1beta1":
			v1beta1 := &pluginServiceV1Beta1{server: e}
			v1beta1.RegisterService()
			break
		case "v1beta2":
			v1beta2 := &pluginServiceV1Beta2{server: e}
			v1beta2.RegisterService()
			break
		default:
			return fmt.Errorf("Unsupported service: '%s'", service)
		}
	}

	// Starts service
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		// Blocking call to accept incoming connections.
		if err := e.grpcServer.Serve(lis); err != nil {
			glog.Errorf("example server stopped serving: %v", err)
		}
	}()

	return nil
}

func (e *examplePlugin) Stop() error {
	glog.Infof("Stopping example server at: %s\n", e.endpoint)

	e.grpcServer.Stop()
	c := make(chan struct{})
	go func() {
		defer close(c)
		e.wg.Wait()
	}()

	select {
	case <-c:
		break
	case <-time.After(time.Second):
		return errors.New("Timed out on waiting for stop completion")
	}

	if err := os.Remove(e.endpoint); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}
