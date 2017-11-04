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
	"fmt"
	neturl "net/url"
	"sync"

	"github.com/golang/glog"
	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"golang.org/x/net/context"
)

// VSphereConnection contains information for connecting to vCenter
type VSphereConnection struct {
	GoVmomiClient     *govmomi.Client
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

// Connect makes connection to vCenter and sets VSphereConnection.GoVmomiClient.
// If connection.GoVmomiClient is already set, it obtains the existing user session.
// if user session is not valid, connection.GoVmomiClient will be set to the new client.
func (connection *VSphereConnection) Connect(ctx context.Context) error {
	var err error
	clientLock.Lock()
	defer clientLock.Unlock()

	if connection.GoVmomiClient == nil {
		connection.GoVmomiClient, err = connection.NewClient(ctx)
		if err != nil {
			glog.Errorf("Failed to create govmomi client. err: %+v", err)
			return err
		}
		return nil
	}
	m := session.NewManager(connection.GoVmomiClient.Client)
	userSession, err := m.UserSession(ctx)
	if err != nil {
		glog.Errorf("Error while obtaining user session. err: %+v", err)
		return err
	}
	if userSession != nil {
		return nil
	}
	glog.Warningf("Creating new client session since the existing session is not valid or not authenticated")
	connection.GoVmomiClient.Logout(ctx)
	connection.GoVmomiClient, err = connection.NewClient(ctx)
	if err != nil {
		glog.Errorf("Failed to create govmomi client. err: %+v", err)
		return err
	}
	return nil
}

// NewClient creates a new govmomi client for the VSphereConnection obj
func (connection *VSphereConnection) NewClient(ctx context.Context) (*govmomi.Client, error) {
	url, err := neturl.Parse(fmt.Sprintf("https://%s:%s/sdk", connection.Hostname, connection.Port))
	if err != nil {
		glog.Errorf("Failed to parse URL: %s. err: %+v", url, err)
		return nil, err
	}
	url.User = neturl.UserPassword(connection.Username, connection.Password)
	client, err := govmomi.NewClient(ctx, url, connection.Insecure)
	if err != nil {
		glog.Errorf("Failed to create new client. err: %+v", err)
		return nil, err
	}
	if connection.RoundTripperCount == 0 {
		connection.RoundTripperCount = RoundTripperDefaultCount
	}
	client.RoundTripper = vim25.Retry(client.RoundTripper, vim25.TemporaryNetworkError(int(connection.RoundTripperCount)))
	return client, nil
}
