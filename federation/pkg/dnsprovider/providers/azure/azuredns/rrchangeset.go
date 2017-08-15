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
	"strings"

	"github.com/Azure/azure-sdk-for-go/arm/dns"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordChangeset = &ResourceRecordChangeset{}

// ResourceRecordChangeset set holds all intended DNS changes (Create, Remove, Upsert).
// Changes are executed by calling the Apply function.
// Changes are submitted one-by-one since the Azure SDK for go doesn't
// offer batching
type ResourceRecordChangeset struct {
	zone   *Zone
	rrsets *ResourceRecordSets

	additions []dnsprovider.ResourceRecordSet
	removals  []dnsprovider.ResourceRecordSet
	upserts   []dnsprovider.ResourceRecordSet
}

// Add adds a ResourceRecordSet to create a new DNS Resource Record
func (c *ResourceRecordChangeset) Add(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.additions = append(c.additions, rrset)
	return c
}

//Remove adds a ResourceRecordSet to remove a DNS Resource Record
func (c *ResourceRecordChangeset) Remove(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.removals = append(c.removals, rrset)
	return c
}

// Upsert adds a ResourceRecordSet to update or create a DNS Resource Record
func (c *ResourceRecordChangeset) Upsert(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.upserts = append(c.upserts, rrset)
	return c
}

// Apply executes all the changes in the changeset
func (c *ResourceRecordChangeset) Apply() error {

	zoneName := c.zone.impl.Name
	// since it looks like the autorest API is request/response we can
	// start with calling the REST APIs one-by-one
	svc := c.rrsets.zone.zones.impl.service

	for _, removal := range c.removals {
		var rset = removal.(ResourceRecordSet).toRecordSet()

		recType := rset.Type

		glog.V(4).Infof("azuredns: Delete:\tRecordSet: %q Type: %q Zone Name: %s TTL: %i ID %q \n", *rset.Name, *recType, *zoneName, *rset.RecordSetProperties.TTL, *rset.ID)
		_, err := svc.DeleteRecordSet(*zoneName, *rset.Name, dns.RecordType(*recType), "")
		if err != nil {
			glog.V(1).Infof("azuredns: Could not delete DNS %s", *rset.Name)
			return err
		}
	}

	for _, upsert := range c.upserts {
		var rset = upsert.(ResourceRecordSet).toRecordSet()

		recType := rset.Type
		glog.V(4).Infof("azuredns: Upsert:\tRecordSet: %s Type: %s Zone Name: %s TTL: %i \n", *rset.Name, *recType, *zoneName, *rset.RecordSetProperties.TTL)

		_, err := svc.CreateOrUpdateRecordSet(*zoneName, *rset.Name, dns.RecordType(*recType), *rset, "", "*")

		if err != nil {
			glog.V(0).Infof("azuredns: Could not upsert DNS %s", upsert.Name)
			return err
		}
	}

	for _, addition := range c.additions {
		var rset = addition.(ResourceRecordSet).toRecordSet()
		recType := rset.Type

		glog.V(4).Infof("azuredns:  Addition:\tRecordSet: %s Type: %s Zone Name: %s TTL: %i \n", *rset.Name, *recType, *zoneName, *rset.RecordSetProperties.TTL)

		props := rset.RecordSetProperties

		if glog.V(5) {
			switch strings.TrimPrefix(*recType, "Microsoft.Network/dnszones/") {
			case "A":
				for i := range *props.ARecords {
					rec := *props.ARecords
					glog.V(0).Infof("azuredns: A Rec Ipv4: %s\n", *rec[i].Ipv4Address)
				}

			case "AAAA":
				for i := range *props.AaaaRecords {
					rec := *props.AaaaRecords
					glog.V(0).Infof("azuredns: AAAA Rec Ipv6: %s\n", *rec[i].Ipv6Address)
				}

			case "CNAME":
				glog.V(5).Infof("azuredns: CNAME: %s for name: %s, ID: %s, TTL %i\n", *props.CnameRecord.Cname, *rset.Name, *rset.ID, *rset.RecordSetProperties.TTL)
			}
		}
		_, err := svc.CreateOrUpdateRecordSet(*zoneName, *rset.Name, dns.RecordType(*recType), *rset, "", "*")
		if err != nil {
			glog.V(0).Infof("azuredns: Could not add DNS %s type %s: %s", addition.Name(), *recType, err.Error())
			return err
		}
	}

	return nil
}

// IsEmpty checks for an empty changeset
func (c *ResourceRecordChangeset) IsEmpty() bool {
	return len(c.removals) == 0 && len(c.additions) == 0 && len(c.upserts) == 0
}

// ResourceRecordSets returns the parent ResourceRecordSets
func (c *ResourceRecordChangeset) ResourceRecordSets() dnsprovider.ResourceRecordSets {
	return c.rrsets
}
