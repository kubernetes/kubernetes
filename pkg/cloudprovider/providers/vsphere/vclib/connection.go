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

package vclib

import (
	"context"
	"net"
	neturl "net/url"
	"sync"

	"github.com/golang/glog"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

// VSphereConnection contains information for connecting to vCenter
type VSphereConnection struct {
	Client            *vim25.Client
	Username          string
	Password          string
	Hostname          string
	Port              string
	Insecure          bool
	RoundTripperCount uint
}

var (
	clientLock sync.Mutex
)

// Connect makes connection to vCenter and sets VSphereConnection.Client.
// If connection.Client is already set, it obtains the existing user session.
// if user session is not valid, connection.Client will be set to the new client.
func (connection *VSphereConnection) Connect(ctx context.Context) error {
	var err error
	clientLock.Lock()
	defer clientLock.Unlock()

	if connection.Client == nil {
		connection.Client, err = connection.NewClient(ctx)
		if err != nil {
			glog.Errorf("Failed to create govmomi client. err: %+v", err)
			return err
		}
		return nil
	}
	m := session.NewManager(connection.Client)
	userSession, err := m.UserSession(ctx)
	if err != nil {
		glog.Errorf("Error while obtaining user session. err: %+v", err)
		return err
	}
	if userSession != nil {
		return nil
	}
	glog.Warningf("Creating new client session since the existing session is not valid or not authenticated")

	connection.Client, err = connection.NewClient(ctx)
	if err != nil {
		glog.Errorf("Failed to create govmomi client. err: %+v", err)
		return err
	}
	return nil
}

// Logout calls SessionManager.Logout for the given connection.
func (connection *VSphereConnection) Logout(ctx context.Context) {
	m := session.NewManager(connection.Client)
	if err := m.Logout(ctx); err != nil {
		glog.Errorf("Logout failed: %s", err)
	}
}

// NewClient creates a new govmomi client for the VSphereConnection obj
func (connection *VSphereConnection) NewClient(ctx context.Context) (*vim25.Client, error) {
	url, err := soap.ParseURL(net.JoinHostPort(connection.Hostname, connection.Port))
	if err != nil {
		glog.Errorf("Failed to parse URL: %s. err: %+v", url, err)
		return nil, err
	}

	sc := soap.NewClient(url, connection.Insecure)
	client, err := vim25.NewClient(ctx, sc)
	if err != nil {
		glog.Errorf("Failed to create new client. err: %+v", err)
		return nil, err
	}

	m := session.NewManager(client)

	err = m.Login(ctx, neturl.UserPassword(connection.Username, connection.Password))
	if err != nil {
		return nil, err
	}

	if connection.RoundTripperCount == 0 {
		connection.RoundTripperCount = RoundTripperDefaultCount
	}
	client.RoundTripper = vim25.Retry(client.RoundTripper, vim25.TemporaryNetworkError(int(connection.RoundTripperCount)))
	return client, nil
}
