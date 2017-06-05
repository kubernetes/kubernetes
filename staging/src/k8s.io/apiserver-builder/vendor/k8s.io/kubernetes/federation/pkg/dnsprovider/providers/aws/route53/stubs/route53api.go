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

/* internal implements a stub for the AWS Route53 API, used primarily for unit testing purposes */
package stubs

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/route53"
)

// Compile time check for interface conformance
var _ Route53API = &Route53APIStub{}

/* Route53API is the subset of the AWS Route53 API that we actually use.  Add methods as required. Signatures must match exactly. */
type Route53API interface {
	ListResourceRecordSetsPages(input *route53.ListResourceRecordSetsInput, fn func(p *route53.ListResourceRecordSetsOutput, lastPage bool) (shouldContinue bool)) error
	ChangeResourceRecordSets(*route53.ChangeResourceRecordSetsInput) (*route53.ChangeResourceRecordSetsOutput, error)
	ListHostedZonesPages(input *route53.ListHostedZonesInput, fn func(p *route53.ListHostedZonesOutput, lastPage bool) (shouldContinue bool)) error
	CreateHostedZone(*route53.CreateHostedZoneInput) (*route53.CreateHostedZoneOutput, error)
	DeleteHostedZone(*route53.DeleteHostedZoneInput) (*route53.DeleteHostedZoneOutput, error)
}

// Route53APIStub is a minimal implementation of Route53API, used primarily for unit testing.
// See http://http://docs.aws.amazon.com/sdk-for-go/api/service/route53.html for descriptions
// of all of its methods.
type Route53APIStub struct {
	zones      map[string]*route53.HostedZone
	recordSets map[string]map[string][]*route53.ResourceRecordSet
}

// NewRoute53APIStub returns an initialized Route53APIStub
func NewRoute53APIStub() *Route53APIStub {
	return &Route53APIStub{
		zones:      make(map[string]*route53.HostedZone),
		recordSets: make(map[string]map[string][]*route53.ResourceRecordSet),
	}
}

func (r *Route53APIStub) ListResourceRecordSetsPages(input *route53.ListResourceRecordSetsInput, fn func(p *route53.ListResourceRecordSetsOutput, lastPage bool) (shouldContinue bool)) error {
	output := route53.ListResourceRecordSetsOutput{} // TODO: Support optional input args.
	if len(r.recordSets) <= 0 {
		output.ResourceRecordSets = []*route53.ResourceRecordSet{}
	} else if _, ok := r.recordSets[*input.HostedZoneId]; !ok {
		output.ResourceRecordSets = []*route53.ResourceRecordSet{}
	} else {
		for _, rrsets := range r.recordSets[*input.HostedZoneId] {
			for _, rrset := range rrsets {
				output.ResourceRecordSets = append(output.ResourceRecordSets, rrset)
			}
		}
	}
	lastPage := true
	fn(&output, lastPage)
	return nil
}

func (r *Route53APIStub) ChangeResourceRecordSets(input *route53.ChangeResourceRecordSetsInput) (*route53.ChangeResourceRecordSetsOutput, error) {
	output := &route53.ChangeResourceRecordSetsOutput{}
	recordSets, ok := r.recordSets[*input.HostedZoneId]
	if !ok {
		recordSets = make(map[string][]*route53.ResourceRecordSet)
	}

	for _, change := range input.ChangeBatch.Changes {
		key := *change.ResourceRecordSet.Name + "::" + *change.ResourceRecordSet.Type
		switch *change.Action {
		case route53.ChangeActionCreate:
			if _, found := recordSets[key]; found {
				return nil, fmt.Errorf("Attempt to create duplicate rrset %s", key) // TODO: Return AWS errors with codes etc
			}
			recordSets[key] = append(recordSets[key], change.ResourceRecordSet)
		case route53.ChangeActionDelete:
			if _, found := recordSets[key]; !found {
				return nil, fmt.Errorf("Attempt to delete non-existent rrset %s", key) // TODO: Check other fields too
			}
			delete(recordSets, key)
		case route53.ChangeActionUpsert:
			// TODO - not used yet
		}
	}
	r.recordSets[*input.HostedZoneId] = recordSets
	return output, nil // TODO: We should ideally return status etc, but we don't' use that yet.
}

func (r *Route53APIStub) ListHostedZonesPages(input *route53.ListHostedZonesInput, fn func(p *route53.ListHostedZonesOutput, lastPage bool) (shouldContinue bool)) error {
	output := &route53.ListHostedZonesOutput{}
	for _, zone := range r.zones {
		output.HostedZones = append(output.HostedZones, zone)
	}
	lastPage := true
	fn(output, lastPage)
	return nil
}

func (r *Route53APIStub) CreateHostedZone(input *route53.CreateHostedZoneInput) (*route53.CreateHostedZoneOutput, error) {
	name := aws.StringValue(input.Name)
	id := "/hostedzone/" + name
	if _, ok := r.zones[id]; ok {
		return nil, fmt.Errorf("Error creating hosted DNS zone: %s already exists", id)
	}
	r.zones[id] = &route53.HostedZone{
		Id:   aws.String(id),
		Name: aws.String(name),
	}
	return &route53.CreateHostedZoneOutput{HostedZone: r.zones[id]}, nil
}

func (r *Route53APIStub) DeleteHostedZone(input *route53.DeleteHostedZoneInput) (*route53.DeleteHostedZoneOutput, error) {
	if _, ok := r.zones[*input.Id]; !ok {
		return nil, fmt.Errorf("Error deleting hosted DNS zone: %s does not exist", *input.Id)
	}
	if len(r.recordSets[*input.Id]) > 0 {
		return nil, fmt.Errorf("Error deleting hosted DNS zone: %s has resource records", *input.Id)
	}
	delete(r.zones, *input.Id)
	return &route53.DeleteHostedZoneOutput{}, nil
}
