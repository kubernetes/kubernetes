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
	"fmt"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordChangeset = &ResourceRecordChangeset{}

type ResourceRecordChangeset struct {
	rrsets *ResourceRecordSets

	additions []dnsprovider.ResourceRecordSet
	removals  []dnsprovider.ResourceRecordSet
}

func (c *ResourceRecordChangeset) Add(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.additions = append(c.additions, rrset)
	return c
}

func (c *ResourceRecordChangeset) Remove(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.removals = append(c.removals, rrset)
	return c
}

func (c *ResourceRecordChangeset) Upsert(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.Remove(rrset)
	c.Add(rrset)
	return c
}

func (c *ResourceRecordChangeset) Apply() error {
	rrsets := c.rrsets

	for _, removal := range c.removals {
		fmt.Println(removal)
		rrset := removal.(ResourceRecordSet)
		records := rrset.records
		for _, record := range records {
			_, err := rrsets.client.Domains.DeleteRecord(rrset.rrsets.zone.ID(), record.ID)
			if err != nil {
				return err
			}
		}
	}

	for _, addition := range c.additions {
		fmt.Println(addition)
		rrset := addition.(ResourceRecordSet)
		records := rrset.records
		for _, record := range records {
			record.Line = "默认"
			_, _, err := rrsets.client.Domains.CreateRecord(rrset.rrsets.zone.ID(), record)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (c *ResourceRecordChangeset) IsEmpty() bool {
	return len(c.removals) == 0 && len(c.additions) == 0
}

// ResourceRecordSets returns the parent ResourceRecordSets
func (c *ResourceRecordChangeset) ResourceRecordSets() dnsprovider.ResourceRecordSets {
	return c.rrsets
}
