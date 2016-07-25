// Copyright 2015 flannel authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This work borrows from the https://github.com/kelseyhightower/flannel-route-manager
// project which has the following license agreement.

// Copyright (c) 2014 Kelsey Hightower

// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package gce

import (
	"fmt"
	"strings"
	"sync"

	log "github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

func init() {
	backend.Register("gce", New)
}

var metadataEndpoint = "http://169.254.169.254/computeMetadata/v1"

var replacer = strings.NewReplacer(".", "-", "/", "-")

type GCEBackend struct {
	sm       subnet.Manager
	extIface *backend.ExternalInterface
	apiInit  sync.Once
	api      *gceAPI
}

func New(sm subnet.Manager, extIface *backend.ExternalInterface) (backend.Backend, error) {
	gb := GCEBackend{
		sm:       sm,
		extIface: extIface,
	}
	return &gb, nil
}

func (g *GCEBackend) ensureAPI() error {
	var err error
	g.apiInit.Do(func() {
		g.api, err = newAPI()
	})
	return err
}

func (g *GCEBackend) Run(ctx context.Context) {
	<-ctx.Done()
}

func (g *GCEBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (backend.Network, error) {
	attrs := subnet.LeaseAttrs{
		PublicIP: ip.FromIP(g.extIface.ExtAddr),
	}

	l, err := g.sm.AcquireLease(ctx, network, &attrs)
	switch err {
	case nil:

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}

	if err = g.ensureAPI(); err != nil {
		return nil, err
	}

	found, err := g.handleMatchingRoute(l.Subnet.String())
	if err != nil {
		return nil, fmt.Errorf("error handling matching route: %v", err)
	}

	if !found {
		operation, err := g.api.insertRoute(l.Subnet.String())
		if err != nil {
			return nil, fmt.Errorf("error inserting route: %v", err)
		}

		err = g.api.pollOperationStatus(operation.Name)
		if err != nil {
			return nil, fmt.Errorf("insert operaiton failed: ", err)
		}
	}

	return &backend.SimpleNetwork{
		SubnetLease: l,
		ExtIface:    g.extIface,
	}, nil
}

//returns true if an exact matching rule is found
func (g *GCEBackend) handleMatchingRoute(subnet string) (bool, error) {
	matchingRoute, err := g.api.getRoute(subnet)
	if err != nil {
		if apiError, ok := err.(*googleapi.Error); ok {
			if apiError.Code != 404 {
				return false, fmt.Errorf("error getting the route err: %v", err)
			}
			return false, nil
		}
		return false, fmt.Errorf("error getting googleapi: %v", err)
	}

	if matchingRoute.NextHopInstance == g.api.gceInstance.SelfLink {
		log.Info("Exact pre-existing route found")
		return true, nil
	}

	log.Info("Deleting conflicting route")
	operation, err := g.api.deleteRoute(subnet)
	if err != nil {
		return false, fmt.Errorf("error deleting conflicting route : %v", err)
	}

	err = g.api.pollOperationStatus(operation.Name)
	if err != nil {
		return false, fmt.Errorf("delete operation failed: %v", err)
	}

	return false, nil
}
