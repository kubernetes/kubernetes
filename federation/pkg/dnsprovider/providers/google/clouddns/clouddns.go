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

	"cloud.google.com/go/compute/metadata"
	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	dns "google.golang.org/api/dns/v1"
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

// getZoneIdMapFromZones takes multiple zones as slice and returns a map with zone names as
// keys and zone id's arranged in slice as values. If multiple zones are passed with same
// name then the zone id's are assigned incrementally starting from 0.
func getZoneIdMapFromZones(zones []string) map[string][]uint64 {
	zoneIdMap := make(map[string][]uint64)
	for _, zoneName := range zones {
		zoneIds := zoneIdMap[zoneName]
		zoneIds = append(zoneIds, uint64(len(zoneIds)))
		zoneIdMap[zoneName] = zoneIds
	}
	return zoneIdMap
}

// NewFakeInterface returns a fake clouddns interface, useful for unit testing purposes.
func NewFakeInterface(zones []string) (dnsprovider.Interface, error) {
	service := stubs.NewService()
	interface_ := newInterfaceWithStub("", service)
	zonesService := service.ManagedZones_
	// Add fake zones to test against.
	zoneIdMap := getZoneIdMapFromZones(zones)
	for zoneName, zoneIds := range zoneIdMap {
		for _, zoneId := range zoneIds {
			zone := &stubs.ManagedZone{Name_: zoneName, Rrsets: []stubs.ResourceRecordSet{}, Id_: zoneId}
			call := zonesService.Create(interface_.project(), zone)
			if _, err := call.Do(); err != nil {
				return nil, err
			}
		}
	}
	return interface_, nil
}
