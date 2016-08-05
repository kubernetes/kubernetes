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

// clouddns is the implementation of pkg/dnsprovider interface for Google Cloud DNS
package clouddns

import (
	"io"

	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	dns "google.golang.org/api/dns/v1"
	"google.golang.org/cloud/compute/metadata"
	gcfg "gopkg.in/gcfg.v1"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/stubs"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
)

const (
	ProviderName = "google-clouddns"
)

func init() {
	dnsprovider.RegisterDnsProvider(ProviderName, func(config io.Reader) (dnsprovider.Interface, error) {
		return newCloudDns(config)
	})
}

type Config struct {
	Global struct {
		TokenURL  string `gcfg:"token-url"`
		TokenBody string `gcfg:"token-body"`
		ProjectID string `gcfg:"project-id"`
	}
}

// newCloudDns creates a new instance of a Google Cloud DNS Interface.
func newCloudDns(config io.Reader) (*Interface, error) {
	projectID, _ := metadata.ProjectID() // On error we get an empty string, which is fine for now.
	var tokenSource oauth2.TokenSource
	// Possibly override defaults with config below
	if config != nil {
		var cfg Config
		if err := gcfg.ReadInto(&cfg, config); err != nil {
			glog.Errorf("Couldn't read config: %v", err)
			return nil, err
		}
		glog.Infof("Using Google Cloud DNS provider config %+v", cfg)
		if cfg.Global.ProjectID != "" {
			projectID = cfg.Global.ProjectID
		}
		if cfg.Global.TokenURL != "" {
			tokenSource = gce.NewAltTokenSource(cfg.Global.TokenURL, cfg.Global.TokenBody)
		}
	}
	return CreateInterface(projectID, tokenSource)
}

// CreateInterface creates a clouddns.Interface object using the specified parameters.
// If no tokenSource is specified, uses oauth2.DefaultTokenSource.
func CreateInterface(projectID string, tokenSource oauth2.TokenSource) (*Interface, error) {
	if tokenSource == nil {
		var err error
		tokenSource, err = google.DefaultTokenSource(
			oauth2.NoContext,
			compute.CloudPlatformScope,
			compute.ComputeScope)
		glog.Infof("Using DefaultTokenSource %#v", tokenSource)
		if err != nil {
			return nil, err
		}
	} else {
		glog.Infof("Using existing Token Source %#v", tokenSource)
	}

	oauthClient := oauth2.NewClient(oauth2.NoContext, tokenSource)

	service, err := dns.New(oauthClient)
	if err != nil {
		glog.Errorf("Failed to get Cloud DNS client: %v", err)
	}
	glog.Infof("Successfully got DNS service: %v\n", service)
	return newInterfaceWithStub(projectID, internal.NewService(service)), nil
}

// NewFakeInterface returns a fake clouddns interface, useful for unit testing purposes.
func NewFakeInterface() (dnsprovider.Interface, error) {
	service := stubs.NewService()
	interface_ := newInterfaceWithStub("", service)
	zones := service.ManagedZones_
	// Add a fake zone to test against.
	zone := &stubs.ManagedZone{Service: zones, Name_: "example.com", Rrsets: []stubs.ResourceRecordSet{}}
	call := zones.Create(interface_.project(), zone)
	if _, err := call.Do(); err != nil {
		return nil, err
	}
	return interface_, nil
}
