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

package route53

import (
	"bytes"
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/route53"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordChangeset = &ResourceRecordChangeset{}

type ResourceRecordChangeset struct {
	zone   *Zone
	rrsets *ResourceRecordSets

	additions []dnsprovider.ResourceRecordSet
	removals  []dnsprovider.ResourceRecordSet
	upserts   []dnsprovider.ResourceRecordSet
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
	c.upserts = append(c.upserts, rrset)
	return c
}

// buildChange converts a dnsprovider.ResourceRecordSet to a route53.Change request
func buildChange(action string, rrs dnsprovider.ResourceRecordSet) *route53.Change {
	change := &route53.Change{
		Action: aws.String(action),
		ResourceRecordSet: &route53.ResourceRecordSet{
			Name: aws.String(rrs.Name()),
			Type: aws.String(string(rrs.Type())),
			TTL:  aws.Int64(rrs.Ttl()),
		},
	}

	for _, rrdata := range rrs.Rrdatas() {
		rr := &route53.ResourceRecord{
			Value: aws.String(rrdata),
		}
		change.ResourceRecordSet.ResourceRecords = append(change.ResourceRecordSet.ResourceRecords, rr)
	}
	return change
}

func (c *ResourceRecordChangeset) Apply() error {
	hostedZoneID := c.zone.impl.Id

	var changes []*route53.Change

	for _, removal := range c.removals {
		change := buildChange(route53.ChangeActionDelete, removal)
		changes = append(changes, change)
	}

	for _, addition := range c.additions {
		change := buildChange(route53.ChangeActionCreate, addition)
		changes = append(changes, change)
	}

	for _, upsert := range c.upserts {
		change := buildChange(route53.ChangeActionUpsert, upsert)
		changes = append(changes, change)
	}

	if len(changes) == 0 {
		return nil
	}

	if glog.V(8) {
		var sb bytes.Buffer
		for _, change := range changes {
			sb.WriteString(fmt.Sprintf("\t%s %s %s\n", aws.StringValue(change.Action), aws.StringValue(change.ResourceRecordSet.Type), aws.StringValue(change.ResourceRecordSet.Name)))
		}

		glog.V(8).Infof("Route53 Changeset:\n%s", sb.String())
	}

	service := c.zone.zones.interface_.service

	request := &route53.ChangeResourceRecordSetsInput{
		ChangeBatch: &route53.ChangeBatch{
			Changes: changes,
		},
		HostedZoneId: hostedZoneID,
	}

	_, err := service.ChangeResourceRecordSets(request)
	if err != nil {
		// Cast err to awserr.Error to get the Code and
		// Message from an error.
		return err
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
