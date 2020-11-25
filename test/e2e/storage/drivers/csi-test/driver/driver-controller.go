/*
Copyright 2019 Kubernetes Authors

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

package driver

import (
	"context"
	"net"
	"sync"

	"google.golang.org/grpc/reflection"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
)

// CSIDriverControllerServer is the Controller service component of the driver.
type CSIDriverControllerServer struct {
	Controller csi.ControllerServer
	Identity   csi.IdentityServer
}

// CSIDriverController is the CSI Driver Controller backend.
type CSIDriverController struct {
	listener         net.Listener
	server           *grpc.Server
	controllerServer *CSIDriverControllerServer
	wg               sync.WaitGroup
	running          bool
	lock             sync.Mutex
	creds            *CSICreds
}

func NewCSIDriverController(controllerServer *CSIDriverControllerServer) *CSIDriverController {
	return &CSIDriverController{
		controllerServer: controllerServer,
	}
}

func (c *CSIDriverController) goServe(started chan<- bool) {
	goServe(c.server, &c.wg, c.listener, started)
}

func (c *CSIDriverController) Address() string {
	return c.listener.Addr().String()
}

func (c *CSIDriverController) Start(l net.Listener) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Set listener.
	c.listener = l

	// Create a new grpc server.
	c.server = grpc.NewServer(
		grpc.UnaryInterceptor(c.callInterceptor),
	)

	if c.controllerServer.Controller != nil {
		csi.RegisterControllerServer(c.server, c.controllerServer.Controller)
	}
	if c.controllerServer.Identity != nil {
		csi.RegisterIdentityServer(c.server, c.controllerServer.Identity)
	}

	reflection.Register(c.server)

	waitForServer := make(chan bool)
	c.goServe(waitForServer)
	<-waitForServer
	c.running = true
	return nil
}

func (c *CSIDriverController) Stop() {
	stop(&c.lock, &c.wg, c.server, c.running)
}

func (c *CSIDriverController) Close() {
	c.server.Stop()
}

func (c *CSIDriverController) IsRunning() bool {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.running
}

func (c *CSIDriverController) SetDefaultCreds() {
	setDefaultCreds(c.creds)
}

func (c *CSIDriverController) callInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	return callInterceptor(ctx, c.creds, req, info, handler)
}
