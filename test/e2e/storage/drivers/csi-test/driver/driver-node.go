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
	context "context"
	"net"
	"sync"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

// CSIDriverNodeServer is the Node service component of the driver.
type CSIDriverNodeServer struct {
	Node     csi.NodeServer
	Identity csi.IdentityServer
}

// CSIDriverNode is the CSI Driver Node backend.
type CSIDriverNode struct {
	listener   net.Listener
	server     *grpc.Server
	nodeServer *CSIDriverNodeServer
	wg         sync.WaitGroup
	running    bool
	lock       sync.Mutex
	creds      *CSICreds
}

func NewCSIDriverNode(nodeServer *CSIDriverNodeServer) *CSIDriverNode {
	return &CSIDriverNode{
		nodeServer: nodeServer,
	}
}

func (c *CSIDriverNode) goServe(started chan<- bool) {
	goServe(c.server, &c.wg, c.listener, started)
}

func (c *CSIDriverNode) Address() string {
	return c.listener.Addr().String()
}

func (c *CSIDriverNode) Start(l net.Listener) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Set listener.
	c.listener = l

	// Create a new grpc server.
	c.server = grpc.NewServer(
		grpc.UnaryInterceptor(c.callInterceptor),
	)

	if c.nodeServer.Node != nil {
		csi.RegisterNodeServer(c.server, c.nodeServer.Node)
	}
	if c.nodeServer.Identity != nil {
		csi.RegisterIdentityServer(c.server, c.nodeServer.Identity)
	}

	reflection.Register(c.server)

	waitForServer := make(chan bool)
	c.goServe(waitForServer)
	<-waitForServer
	c.running = true
	return nil
}

func (c *CSIDriverNode) Stop() {
	stop(&c.lock, &c.wg, c.server, c.running)
}

func (c *CSIDriverNode) Close() {
	c.server.Stop()
}

func (c *CSIDriverNode) IsRunning() bool {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.running
}

func (c *CSIDriverNode) SetDefaultCreds() {
	setDefaultCreds(c.creds)
}

func (c *CSIDriverNode) callInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	return callInterceptor(ctx, c.creds, req, info, handler)
}
