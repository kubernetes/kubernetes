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

package clouddns

import (
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone *Zone
	impl interfaces.ResourceRecordSetsService
}

func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	response, err := rrsets.impl.List(rrsets.project(), rrsets.zone.impl.Name()).Do()
	if err != nil {
		return nil, err
	}
	list := make([]dnsprovider.ResourceRecordSet, len(response.Rrsets()))
	for i, rrset := range response.Rrsets() {
		list[i] = ResourceRecordSet{rrset, &rrsets}
	}
	return list, nil
}

func (rrsets ResourceRecordSets) Get(name string) (dnsprovider.ResourceRecordSet, error) {
	var newRrset dnsprovider.ResourceRecordSet
	rrsetList, err := rrsets.List()
	if err != nil {
		return nil, err
	}
	for _, rrset := range rrsetList {
		if rrset.Name() == name {
			newRrset = rrset
			break
		}
	}
	return newRrset, nil
}

func (r ResourceRecordSets) StartChangeset() dnsprovider.ResourceRecordChangeset {
	return &ResourceRecordChangeset{
		rrsets: &r,
	}
}

func (r ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	return ResourceRecordSet{r.impl.NewResourceRecordSet(name, rrdatas, ttl, rrstype), &r}
}

func (rrsets ResourceRecordSets) project() string {
	return rrsets.zone.project()
}

// Zone returns the parent zone
func (rrset ResourceRecordSets) Zone() dnsprovider.Zone {
	return rrset.zone
}
