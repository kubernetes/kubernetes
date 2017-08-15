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
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSet = ResourceRecordSet{}

// ResourceRecordSet implements the federation interface
// dnsprovider.ResourceRecordSet.
// The struct holds the Azure DNS implmentation of the corresponding RecordSet
// It also allows navigation of the DNS hierarchy via ResourceRecordSet -> ResourceRecordSets -> Zone -> Zones
type ResourceRecordSet struct {
	impl   *dns.RecordSet
	rrsets *ResourceRecordSets
}

// Name returns the absolute ResourceRecordSet name, i.e. the name includes the zone
func (rrset ResourceRecordSet) Name() string {
	if *rrset.impl.Name == "@" {
		return *rrset.impl.Name
	}

	// k8s wants the full name, not the relative name
	// the Azure DNS Recordset only has the relative name. Add the zone without the training dot
	return *rrset.impl.Name + "." + rrset.rrsets.zone.Name()
}

// Rrdatas returns the record set details in string[] format.
func (rrset ResourceRecordSet) Rrdatas() []string {
	return rrset.getRrDatas()
}

// Ttl returns the RecordSets time-to-live
func (rrset ResourceRecordSet) Ttl() int64 {
	// same behavior as the route 53 provider
	if rrset.impl.TTL != nil {
		return *rrset.impl.TTL
	}
	return 0
}

// Type returns the DNS record type of this ResourceRecordSet
func (rrset ResourceRecordSet) Type() rrstype.RrsType {
	// Azure DNS API prefixes the type with Microsoft.Network/dnszones/.
	// k8s expects only the DNS record type
	return rrstype.RrsType(strings.TrimPrefix(*rrset.impl.Type, "Microsoft.Network/dnszones/"))
}

func (rrset ResourceRecordSet) toRecordSet() *dns.RecordSet {
	recType := string(rrset.Type())
	// make sure to use the relative name of the RecordSet
	nameCopy := string([]byte(*rrset.impl.Name))

	r := &dns.RecordSet{
		Name: &nameCopy,
		Type: to.StringPtr(recType),
		ID:   &nameCopy,
	}

	glog.V(5).Infof("New RecordSet: Name: %s ID: %s, Type: %s\n", *r.Name, *r.ID, *r.Type)

	addRrDatasToRecordSet(r, rrset.Rrdatas())
	r.RecordSetProperties.TTL = to.Int64Ptr(rrset.Ttl())
	return r
}

func (rrset ResourceRecordSet) getRrDatas() []string {

	props := rrset.impl.RecordSetProperties
	var rrDatas []string

	switch rrset.Type() {
	case "A":
		rrDatas = make([]string, len(*props.ARecords))

		for i := range *props.ARecords {
			rec := *props.ARecords
			rrDatas[i] = *rec[i].Ipv4Address
		}

	case "AAAA":
		rrDatas = make([]string, len(*props.AaaaRecords))

		for i := range *props.AaaaRecords {
			rec := *props.AaaaRecords
			rrDatas[i] = *rec[i].Ipv6Address
		}

	case "CNAME":
		rrDatas = make([]string, 1)
		rrDatas[0] = *props.CnameRecord.Cname
	}

	return rrDatas
}

func addRrDatasToRecordSet(rs *dns.RecordSet, rrDatas []string) {
	props := dns.RecordSetProperties{}
	var i int
	rrsType := string(*rs.Type)
	// kubernetes 1.6.2 only handles A, AAAA and CNAME
	switch rrsType {
	case "A":
		recs := make([]dns.ARecord, 0)

		rrmap := make(map[string]string)

		for i = range rrDatas {
			if _, ok := rrmap[rrDatas[i]]; !ok {
				rrmap[rrDatas[i]] = rrDatas[i]
				recs = append(recs, dns.ARecord{
					Ipv4Address: to.StringPtr(rrDatas[i]),
				})
			}
		}
		props.ARecords = &recs

	case "AAAA":
		recs := make([]dns.AaaaRecord, len(rrDatas))
		for i = range rrDatas {
			recs[i] = dns.AaaaRecord{
				Ipv6Address: to.StringPtr(rrDatas[i]),
			}
		}
		props.AaaaRecords = &recs

	case "CNAME":
		for i = range rrDatas {
			props.CnameRecord = &dns.CnameRecord{
				Cname: to.StringPtr(rrDatas[i]),
			}
		}
	}

	rs.RecordSetProperties = &props
}

func (rrset ResourceRecordSet) setRecordSetProperties(ttl int64, rrDatas []string) dnsprovider.ResourceRecordSet {

	addRrDatasToRecordSet(rrset.impl, rrDatas)
	rrset.impl.RecordSetProperties.TTL = to.Int64Ptr(ttl)

	return rrset
}
