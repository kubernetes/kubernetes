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

package dnspod

import (
	"strconv"
	"strings"

	dns "github.com/decker502/dnspod-go"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone   *Zone
	client *dns.Client
}

type recordNameType struct {
	recordName string
	recordType string
}

func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	records, _, err := rrsets.client.Domains.ListRecords(rrsets.zone.ID(), "")
	if err != nil {
		return nil, err
	}

	maps := map[recordNameType][]dns.Record{}
	for _, record := range records {
		maps[recordNameType{record.Name, record.Type}] = append(maps[recordNameType{record.Name, record.Type}], record)
	}

	list := make([]dnsprovider.ResourceRecordSet, len(maps))
	i := 0
	for p := range maps {
		list[i] = ResourceRecordSet{rrsets.zone.domain, maps[p], &rrsets}
		i = i + 1
	}

	return list, nil
}

func (rrsets ResourceRecordSets) Get(name string) ([]dnsprovider.ResourceRecordSet, error) {
	rrsetList, err := rrsets.List()
	if err != nil {
		return nil, err
	}

	var list []dnsprovider.ResourceRecordSet
	for _, rrset := range rrsetList {
		if rrset.Name() == name {
			list = append(list, rrset)
		}
	}
	return list, nil
}

func (r ResourceRecordSets) StartChangeset() dnsprovider.ResourceRecordChangeset {
	return &ResourceRecordChangeset{
		rrsets: &r,
	}
}

func (r ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	domain := dns.Domain{Name: r.zone.Name(), TTL: strconv.FormatInt(ttl, 10)}
	records := make([]dns.Record, len(rrdatas))
	for i, rrdata := range rrdatas {
		records[i] = dns.Record{Name: strings.TrimRight(name, domain.Name), Value: rrdata, Type: string(rrstype), TTL: domain.TTL}
	}
	return ResourceRecordSet{domain, records, &r}
}

// Zone returns the parent zone
func (rrset ResourceRecordSets) Zone() dnsprovider.Zone {
	return rrset.zone
}
