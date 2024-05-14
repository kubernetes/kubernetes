/*
Copyright 2021 The Kubernetes Authors.

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

//go:generate mockgen -package=driver -destination=driver.mock.go -build_flags=-mod=readonly github.com/container-storage-interface/spec/lib/go/csi IdentityServer,ControllerServer,NodeServer

package driver

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
)

var (
	// ErrNoCredentials is the error when a secret is enabled but not passed in the request.
	ErrNoCredentials = errors.New("secret must be provided")
	// ErrAuthFailed is the error when the secret is incorrect.
	ErrAuthFailed = errors.New("authentication failed")
)

// CSIDriverServers is a unified driver component with both Controller and Node
// services.
type CSIDriverServers struct {
	Controller csi.ControllerServer
	Identity   csi.IdentityServer
	Node       csi.NodeServer
}

// This is the key name in all the CSI secret objects.
const secretField = "secretKey"

// CSICreds is a driver specific secret type. Drivers can have a key-val pair of
// secrets. This mock driver has a single string secret with secretField as the
// key.
type CSICreds struct {
	CreateVolumeSecret                         string
	DeleteVolumeSecret                         string
	ControllerPublishVolumeSecret              string
	ControllerUnpublishVolumeSecret            string
	NodeStageVolumeSecret                      string
	NodePublishVolumeSecret                    string
	CreateSnapshotSecret                       string
	DeleteSnapshotSecret                       string
	ControllerValidateVolumeCapabilitiesSecret string
}

type CSIDriver struct {
	listener net.Listener
	server   *grpc.Server
	servers  *CSIDriverServers
	wg       sync.WaitGroup
	running  bool
	lock     sync.Mutex
	creds    *CSICreds
	logGRPC  LogGRPC
}

type LogGRPC func(method string, request, reply interface{}, err error)

func NewCSIDriver(servers *CSIDriverServers) *CSIDriver {
	return &CSIDriver{
		servers: servers,
	}
}

func (c *CSIDriver) goServe(started chan<- bool) {
	goServe(c.server, &c.wg, c.listener, started)
}

func (c *CSIDriver) Address() string {
	return c.listener.Addr().String()
}

// Start runs a gRPC server with all enabled services. If an interceptor
// is give, then it will be used. Otherwise, an interceptor which
// handles simple credential checks and logs gRPC calls in JSON format
// will be used.
func (c *CSIDriver) Start(l net.Listener, interceptor grpc.UnaryServerInterceptor) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Set listener
	c.listener = l

	// Create a new grpc server
	if interceptor == nil {
		interceptor = c.callInterceptor
	}
	c.server = grpc.NewServer(grpc.UnaryInterceptor(interceptor))

	// Register Mock servers
	if c.servers.Controller != nil {
		csi.RegisterControllerServer(c.server, c.servers.Controller)
	}
	if c.servers.Identity != nil {
		csi.RegisterIdentityServer(c.server, c.servers.Identity)
	}
	if c.servers.Node != nil {
		csi.RegisterNodeServer(c.server, c.servers.Node)
	}

	// Start listening for requests
	waitForServer := make(chan bool)
	c.goServe(waitForServer)
	<-waitForServer
	c.running = true
	return nil
}

func (c *CSIDriver) Stop() {
	stop(&c.lock, &c.wg, c.server, c.running)
}

func (c *CSIDriver) Close() {
	c.server.Stop()
}

func (c *CSIDriver) IsRunning() bool {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.running
}

// SetDefaultCreds sets the default secrets for CSI creds.
func (c *CSIDriver) SetDefaultCreds() {
	setDefaultCreds(c.creds)
}

// goServe starts a grpc server.
func goServe(server *grpc.Server, wg *sync.WaitGroup, listener net.Listener, started chan<- bool) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		started <- true
		err := server.Serve(listener)
		if err != nil {
			klog.Infof("gRPC server for CSI driver stopped: %v", err)
		}
	}()
}

// stop stops a grpc server.
func stop(lock *sync.Mutex, wg *sync.WaitGroup, server *grpc.Server, running bool) {
	lock.Lock()
	defer lock.Unlock()

	if !running {
		return
	}

	server.Stop()
	wg.Wait()
}

// setDefaultCreds sets the default credentials, given a CSICreds instance.
func setDefaultCreds(creds *CSICreds) {
	*creds = CSICreds{
		CreateVolumeSecret:                         "secretval1",
		DeleteVolumeSecret:                         "secretval2",
		ControllerPublishVolumeSecret:              "secretval3",
		ControllerUnpublishVolumeSecret:            "secretval4",
		NodeStageVolumeSecret:                      "secretval5",
		NodePublishVolumeSecret:                    "secretval6",
		CreateSnapshotSecret:                       "secretval7",
		DeleteSnapshotSecret:                       "secretval8",
		ControllerValidateVolumeCapabilitiesSecret: "secretval9",
	}
}

func (c *CSIDriver) callInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	err := authInterceptor(c.creds, req)
	if err != nil {
		logGRPC(info.FullMethod, req, nil, err)
		return nil, err
	}
	rsp, err := handler(ctx, req)
	logGRPC(info.FullMethod, req, rsp, err)
	if c.logGRPC != nil {
		c.logGRPC(info.FullMethod, req, rsp, err)
	}
	return rsp, err
}

func authInterceptor(creds *CSICreds, req interface{}) error {
	if creds != nil {
		authenticated, authErr := isAuthenticated(req, creds)
		if !authenticated {
			if authErr == ErrNoCredentials {
				return status.Error(codes.InvalidArgument, authErr.Error())
			}
			if authErr == ErrAuthFailed {
				return status.Error(codes.Unauthenticated, authErr.Error())
			}
		}
	}
	return nil
}

func logGRPC(method string, request, reply interface{}, err error) {
	// Log JSON with the request and response for easier parsing
	logMessage := struct {
		Method   string
		Request  interface{}
		Response interface{}
		// Error as string, for backward compatibility.
		// "" on no error.
		Error string
		// Full error dump, to be able to parse out full gRPC error code and message separately in a test.
		FullError error
	}{
		Method:    method,
		Request:   request,
		Response:  reply,
		FullError: err,
	}

	if err != nil {
		logMessage.Error = err.Error()
	}

	msg, _ := json.Marshal(logMessage)
	klog.V(3).Infof("gRPCCall: %s\n", msg)
}

func isAuthenticated(req interface{}, creds *CSICreds) (bool, error) {
	switch r := req.(type) {
	case *csi.CreateVolumeRequest:
		return authenticateCreateVolume(r, creds)
	case *csi.DeleteVolumeRequest:
		return authenticateDeleteVolume(r, creds)
	case *csi.ControllerPublishVolumeRequest:
		return authenticateControllerPublishVolume(r, creds)
	case *csi.ControllerUnpublishVolumeRequest:
		return authenticateControllerUnpublishVolume(r, creds)
	case *csi.NodeStageVolumeRequest:
		return authenticateNodeStageVolume(r, creds)
	case *csi.NodePublishVolumeRequest:
		return authenticateNodePublishVolume(r, creds)
	case *csi.CreateSnapshotRequest:
		return authenticateCreateSnapshot(r, creds)
	case *csi.DeleteSnapshotRequest:
		return authenticateDeleteSnapshot(r, creds)
	case *csi.ValidateVolumeCapabilitiesRequest:
		return authenticateControllerValidateVolumeCapabilities(r, creds)
	default:
		return true, nil
	}
}

func authenticateCreateVolume(req *csi.CreateVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.CreateVolumeSecret)
}

func authenticateDeleteVolume(req *csi.DeleteVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.DeleteVolumeSecret)
}

func authenticateControllerPublishVolume(req *csi.ControllerPublishVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.ControllerPublishVolumeSecret)
}

func authenticateControllerUnpublishVolume(req *csi.ControllerUnpublishVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.ControllerUnpublishVolumeSecret)
}

func authenticateNodeStageVolume(req *csi.NodeStageVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.NodeStageVolumeSecret)
}

func authenticateNodePublishVolume(req *csi.NodePublishVolumeRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.NodePublishVolumeSecret)
}

func authenticateCreateSnapshot(req *csi.CreateSnapshotRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.CreateSnapshotSecret)
}

func authenticateDeleteSnapshot(req *csi.DeleteSnapshotRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.DeleteSnapshotSecret)
}

func authenticateControllerValidateVolumeCapabilities(req *csi.ValidateVolumeCapabilitiesRequest, creds *CSICreds) (bool, error) {
	return credsCheck(req.GetSecrets(), creds.ControllerValidateVolumeCapabilitiesSecret)
}

func credsCheck(secrets map[string]string, secretVal string) (bool, error) {
	if len(secrets) == 0 {
		return false, ErrNoCredentials
	}

	if secrets[secretField] != secretVal {
		return false, ErrAuthFailed
	}
	return true, nil
}
