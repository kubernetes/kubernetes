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

package vsphere

import (
	"context"
	"fmt"
	neturl "net/url"
	"sync"

	"github.com/golang/glog"
	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
)

const (
	roundTripperDefaultCount = 3
)

var (
	clientLock sync.Mutex
)

// Connect makes connection to vSphere
// No actions are taken if a connection exists and alive. Otherwise, a new client will be created.
func Connect(ctx context.Context, vs *VSphere) error {
	var err error
	clientLock.Lock()
	defer clientLock.Unlock()

	if vs.Client == nil {
		vs.Client, err = NewClient(ctx, vs)
		if err != nil {
			glog.Errorf("Failed to create govmomi client. err: %+v", err)
			return err
		}
		return nil
	}
	manager := session.NewManager(vs.Client.Client)
	userSession, err := manager.UserSession(ctx)
	if err != nil {
		glog.Errorf("Error while obtaining user session. err: %+v", err)
		return err
	}
	if userSession != nil {
		return nil
	}
	glog.Warningf("Creating new client session since the existing session is not valid or not authenticated")
	vs.Client.Logout(ctx)
	vs.Client, err = NewClient(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to create govmomi client. err: %+v", err)
		return err
	}
	return nil
}

// NewClient creates a new client for vSphere connection
func NewClient(ctx context.Context, vs *VSphere) (*govmomi.Client, error) {
	url, err := neturl.Parse(fmt.Sprintf("https://%s:%s/sdk", vs.Config.Hostname, vs.Config.Port))
	if err != nil {
		glog.Errorf("Failed to parse URL: %s. err: %+v", url, err)
		return nil, err
	}
	url.User = neturl.UserPassword(vs.Config.Username, vs.Config.Password)
	client, err := govmomi.NewClient(ctx, url, true)
	if err != nil {
		glog.Errorf("Failed to create new client. err: %+v", err)
		return nil, err
	}
	if vs.Config.RoundTripperCount == 0 {
		vs.Config.RoundTripperCount = roundTripperDefaultCount
	}
	client.RoundTripper = vim25.Retry(client.RoundTripper, vim25.TemporaryNetworkError(int(vs.Config.RoundTripperCount)))
	return client, nil
}
