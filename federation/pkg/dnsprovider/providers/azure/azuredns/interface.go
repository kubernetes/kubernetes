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

package azuredns

import (
	"fmt"

	"github.com/Azure/azure-sdk-for-go/arm/dns"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	azurestub "k8s.io/kubernetes/federation/pkg/dnsprovider/providers/azure/azuredns/stubs"
)

// API is an interface abstracting the Azure DNS clients from azure-sdk-for-go behind a single interface
// The mock implementation is below
// The interface is defined in stubs/azurednsapi.go

// Compile time check for interface adherence
var _ dnsprovider.Interface = Interface{}

// Interface is the abstraction layer to allow for mocking
type Interface struct {
	service azurestub.API
}

// Zones initializes a new Zones interface, which is the root
// of the DNS hierarchy
func (c Interface) Zones() (dnsprovider.Zones, bool) {
	return Zones{&c}, true
}

// compile time check
var _ azurestub.API = &DNSAPI{}

// DNSAPI implements the API interface, which abstracts a small subset of
// the Azure SDK DNS API for mocking purposes
type DNSAPI struct {
	rc   dns.RecordSetsClient
	zc   dns.ZonesClient
	conf Config
}

// DeleteRecordSet deletes a DNS record
func (c *DNSAPI) DeleteRecordSet(zoneName string, relativeRecordSetName string, recordType dns.RecordType, ifMatch string) (result autorest.Response, err error) {
	glog.V(4).Infof("azuredns: Deleting RecordSet %q type %q for zone %s in rg %q\n", relativeRecordSetName, string(recordType), zoneName, c.conf.Global.ResourceGroup)

	return c.rc.Delete(c.conf.Global.ResourceGroup, zoneName, relativeRecordSetName, recordType, ifMatch)
}

// CreateOrUpdateRecordSet creates or updates a Record Set
func (c *DNSAPI) CreateOrUpdateRecordSet(zoneName string, relativeRecordSetName string, recordType dns.RecordType, parameters dns.RecordSet, ifMatch string, ifNoneMatch string) (dns.RecordSet, error) {
	glog.V(4).Infof("azuredns: CreateOrUpdate RecordSets %q type %q for zone %q in rg %q\n", relativeRecordSetName, string(recordType), zoneName, c.conf.Global.ResourceGroup)

	return c.rc.CreateOrUpdate(c.conf.Global.ResourceGroup,
		zoneName, relativeRecordSetName, recordType, parameters, ifMatch, ifNoneMatch)
}

func (c *DNSAPI) appendListRecordSetsResult(rrsets *[]dns.RecordSet, result dns.RecordSetListResult) error {
	for _, rset := range *result.Value {
		*rrsets = append(*rrsets, rset)
	}

	if result.NextLink != nil {
		result, err := c.rc.ListByDNSZoneNextResults(result)
		if err == nil {
			c.appendListRecordSetsResult(rrsets, result)
		} else {
			return err
		}
	}
	return nil
}

// ListResourceRecordSetsByZone lists all record sets for a zone
func (c *DNSAPI) ListResourceRecordSetsByZone(zoneName string) (*[]dns.RecordSet, error) {
	glog.V(5).Infof("azuredns: Listing RecordSets for zone %s in rg %s\n", zoneName, c.conf.Global.ResourceGroup)

	rrsets := make([]dns.RecordSet, 0)

	result, err := c.rc.ListByDNSZone(c.conf.Global.ResourceGroup,
		zoneName,
		to.Int32Ptr(1000))

	err = c.appendListRecordSetsResult(&rrsets, result)

	if err != nil {
		return nil, err
	}
	return &rrsets, nil
}

// ListZones lists the zones in the configured resource group
func (c *DNSAPI) ListZones() (dns.ZoneListResult, error) {
	glog.V(5).Infof("azuredns: Requesting DNS zones")
	// request all 100 zones. 100 is the current limit per subscription
	return c.zc.List(to.Int32Ptr(100))
}

// CreateOrUpdateZone creates or updates a zone
func (c *DNSAPI) CreateOrUpdateZone(zoneName string, zone dns.Zone, ifMatch string, ifNoneMatch string) (dns.Zone, error) {
	glog.V(4).Infof("azuredns: Creating Zone: %s, in resource group: %s\n", zoneName, c.conf.Global.ResourceGroup)
	return c.zc.CreateOrUpdate(c.conf.Global.ResourceGroup, zoneName, zone, ifMatch, ifNoneMatch)
}

// DeleteZone deletes a Zone from the configured Azure resource group
func (c *DNSAPI) DeleteZone(zoneName string, ifMatch string, cancel <-chan struct{}) (<-chan dns.ZoneDeleteResult, <-chan error) {
	glog.V(4).Infof("azuredns: Removing Azure DNS zone Name: %s rg: %s\n", zoneName, c.conf.Global.ResourceGroup)
	return c.zc.Delete(c.conf.Global.ResourceGroup, zoneName, ifMatch, cancel)
}

// New initializes a new API interface from the --dns-provider-config
// The --dns-provider-config option is required.
// In the future, we could try inferring defaults.
func New(config Config) *Interface {
	api := &DNSAPI{}

	glog.V(4).Infof("azuredns: Created Azure DNS DNSAPI for subscription: %s", config.Global.SubscriptionID)
	api.conf = config

	api.zc = dns.NewZonesClient(config.Global.SubscriptionID)
	spt, err := NewServicePrincipalTokenFromCredentials(config, azure.PublicCloud.ResourceManagerEndpoint)
	if err != nil {
		glog.Fatalf("azuredns: Error authenticating to Azure DNS: %v", err)
		return nil
	}

	api.zc.Authorizer = autorest.NewBearerAuthorizer(spt)

	api.rc = dns.NewRecordSetsClient(config.Global.SubscriptionID)
	spt, err = NewServicePrincipalTokenFromCredentials(config, azure.PublicCloud.ResourceManagerEndpoint)
	if err != nil {
		glog.Fatalf("azuredns: Error authenticating to Azure DNS: %v", err)
		return nil
	}

	api.rc.Authorizer = autorest.NewBearerAuthorizer(spt)
	return &Interface{service: api}
}

func checkEnvVar(envVars *map[string]string) error {
	var missingVars []string
	for varName, value := range *envVars {
		if value == "" {
			missingVars = append(missingVars, varName)
		}
	}
	if len(missingVars) > 0 {
		return fmt.Errorf("azuredns: Missing environment variables %v", missingVars)
	}
	return nil
}
