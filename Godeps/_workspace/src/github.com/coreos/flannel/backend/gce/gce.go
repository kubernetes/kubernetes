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
	"net"
	"strings"
	"time"

	"github.com/coreos/flannel/Godeps/_workspace/src/code.google.com/p/goauth2/compute/serviceaccount"
	"github.com/coreos/flannel/Godeps/_workspace/src/code.google.com/p/google-api-go-client/compute/v1"
	"github.com/coreos/flannel/Godeps/_workspace/src/code.google.com/p/google-api-go-client/googleapi"
	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

var metadataEndpoint = "http://169.254.169.254/computeMetadata/v1"

var replacer = strings.NewReplacer(".", "-", "/", "-")

type GCEBackend struct {
	sm             subnet.Manager
	publicIP       ip.IP4
	mtu            int
	project        string
	lease          *subnet.Lease
	computeService *compute.Service
	gceNetwork     *compute.Network
	gceInstance    *compute.Instance
}

func New(sm subnet.Manager, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error) {
	gb := GCEBackend{
		sm:       sm,
		publicIP: ip.FromIP(extEaddr),
		mtu:      extIface.MTU,
	}
	return &gb, nil
}

func (g *GCEBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (*backend.SubnetDef, error) {
	attrs := subnet.LeaseAttrs{
		PublicIP: g.publicIP,
	}

	l, err := g.sm.AcquireLease(ctx, network, &attrs)
	switch err {
	case nil:
		g.lease = l

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}

	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err)
	}

	g.computeService, err = compute.New(client)
	if err != nil {
		return nil, fmt.Errorf("error creating compute service: %v", err)
	}

	networkName, err := networkFromMetadata()
	if err != nil {
		return nil, fmt.Errorf("error getting network metadata: %v", err)
	}

	g.project, err = projectFromMetadata()
	if err != nil {
		return nil, fmt.Errorf("error getting project: %v", err)
	}

	instanceName, err := instanceNameFromMetadata()
	if err != nil {
		return nil, fmt.Errorf("error getting instance name: %v", err)
	}

	instanceZone, err := instanceZoneFromMetadata()
	if err != nil {
		return nil, fmt.Errorf("error getting instance zone: %v", err)
	}

	g.gceNetwork, err = g.computeService.Networks.Get(g.project, networkName).Do()
	if err != nil {
		return nil, fmt.Errorf("error getting network from compute service: %v", err)
	}

	g.gceInstance, err = g.computeService.Instances.Get(g.project, instanceZone, instanceName).Do()
	if err != nil {
		return nil, fmt.Errorf("error getting instance from compute service: %v", err)
	}

	found, err := g.handleMatchingRoute(l.Subnet.String())
	if err != nil {
		return nil, fmt.Errorf("error handling matching route: %v", err)
	}

	if !found {
		operation, err := g.insertRoute(l.Subnet.String())
		if err != nil {
			return nil, fmt.Errorf("error inserting route: %v", err)
		}

		err = g.pollOperationStatus(operation.Name)
		if err != nil {
			return nil, fmt.Errorf("insert operaiton failed: ", err)
		}
	}

	return &backend.SubnetDef{
		Lease: l,
		MTU:   g.mtu,
	}, nil
}

func (g *GCEBackend) Run(ctx context.Context) {
}

func (g *GCEBackend) pollOperationStatus(operationName string) error {
	for i := 0; i < 100; i++ {
		operation, err := g.computeService.GlobalOperations.Get(g.project, operationName).Do()
		if err != nil {
			return fmt.Errorf("error fetching operation status: %v", err)
		}

		if operation.Error != nil {
			return fmt.Errorf("error running operation: %v", operation.Error)
		}

		if i%5 == 0 {
			log.Infof("%v operation status: %v waiting for completion...", operation.OperationType, operation.Status)
		}

		if operation.Status == "DONE" {
			return nil
		}
		time.Sleep(time.Second)
	}

	return fmt.Errorf("timeout waiting for operation to finish")
}

//returns true if an exact matching rule is found
func (g *GCEBackend) handleMatchingRoute(subnet string) (bool, error) {
	matchingRoute, err := g.getRoute(subnet)
	if err != nil {
		if apiError, ok := err.(*googleapi.Error); ok {
			if apiError.Code != 404 {
				return false, fmt.Errorf("error getting the route err: %v", err)
			}
			return false, nil
		}
		return false, fmt.Errorf("error getting googleapi: %v", err)
	}

	if matchingRoute.NextHopInstance == g.gceInstance.SelfLink {
		log.Info("Exact pre-existing route found")
		return true, nil
	}

	log.Info("Deleting conflicting route")
	operation, err := g.deleteRoute(subnet)
	if err != nil {
		return false, fmt.Errorf("error deleting conflicting route : %v", err)
	}

	err = g.pollOperationStatus(operation.Name)
	if err != nil {
		return false, fmt.Errorf("delete operation failed: %v", err)
	}

	return false, nil

}

func (g *GCEBackend) getRoute(subnet string) (*compute.Route, error) {
	routeName := formatRouteName(subnet)
	return g.computeService.Routes.Get(g.project, routeName).Do()
}

func (g *GCEBackend) deleteRoute(subnet string) (*compute.Operation, error) {
	routeName := formatRouteName(subnet)
	return g.computeService.Routes.Delete(g.project, routeName).Do()
}

func (g *GCEBackend) insertRoute(subnet string) (*compute.Operation, error) {
	log.Infof("Inserting route for subnet: %v", subnet)
	route := &compute.Route{
		Name:            formatRouteName(subnet),
		DestRange:       subnet,
		Network:         g.gceNetwork.SelfLink,
		NextHopInstance: g.gceInstance.SelfLink,
		Priority:        1000,
		Tags:            []string{},
	}
	return g.computeService.Routes.Insert(g.project, route).Do()
}

func formatRouteName(subnet string) string {
	return fmt.Sprintf("flannel-%s", replacer.Replace(subnet))
}
