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

package interfaces

import (
	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Interfaces to directly mirror the Google Cloud DNS API structures.
// See https://godoc.org/google.golang.org/api/dns/v1 for details
// This facilitates stubbing out Google Cloud DNS for unit testing.
// Only the parts of the API that we use are included.
// Others can be added as needed.

type (
	Change interface {
		Additions() []ResourceRecordSet
		Deletions() []ResourceRecordSet
		// Id() string  // TODO: Add as needed
		// Kind() string // TODO: Add as needed
		// StartTime() string // TODO: Add as needed
		// Status() string // TODO: Add as needed
	}

	ChangesCreateCall interface {
		// Context(ctx context.Context) *ChangesCreateCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (Change, error)
		// Fields(s ...googleapi.Field) *ChangesCreateCall // TODO: Add as needed
	}

	ChangesGetCall interface {
		// Context(ctx context.Context) *ChangesGetCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (*Change, error)
		// Fields(s ...googleapi.Field) *ChangesGetCall // TODO: Add as needed
		// IfNoneMatch(entityTag string) *ChangesGetCall // TODO: Add as needed
	}

	ChangesListCall interface {
		// Context(ctx context.Context) *ChangesListCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (*ChangesListResponse, error)
		// Fields(s ...googleapi.Field) *ChangesListCall // TODO: Add as needed
		// IfNoneMatch(entityTag string) *ChangesListCall // TODO: Add as needed
		// MaxResults(maxResults int64) *ChangesListCall // TODO: Add as needed
		// PageToken(pageToken string) *ChangesListCall // TODO: Add as needed
		// Pages(ctx context.Context, f func(*ChangesListResponse) error) error // TODO: Add as needed
		// SortBy(sortBy string) *ChangesListCall // TODO: Add as needed
		// SortOrder(sortOrder string) *ChangesListCall // TODO: Add as needed
	}

	ChangesListResponse interface {
		// Changes() []*Change // TODO: Add as needed
		// Kind() string // TODO: Add as needed
		// NextPageToken() string // TODO: Add as needed
		// ServerResponse() googleapi.ServerResponse // TODO: Add as needed
		// ForceSendFields() []string // TODO: Add as needed
	}

	ChangesService interface {
		// Create(project string, managedZone string, change *Change) *ChangesCreateCall // TODO: Add as needed
		Create(project string, managedZone string, change Change) ChangesCreateCall
		NewChange(additions, deletions []ResourceRecordSet) Change

		// Get(project string, managedZone string, changeId string) *ChangesGetCall // TODO: Add as needed
		// List(project string, managedZone string) *ChangesListCall // TODO: Add as needed
	}

	ManagedZone interface {
		// CreationTime() string // TODO: Add as needed
		// Description() string // TODO: Add as needed
		DnsName() string
		// Id()  uint64 // TODO: Add as needed
		// Kind() string // TODO: Add as needed
		Name() string
		// NameServerSet() string // TODO: Add as needed
		// NameServers() []string // TODO: Add as needed
		// ServerResponse() googleapi.ServerResponse // TODO: Add as needed
		// ForceSendFields() []string // TODO: Add as needed
	}

	ManagedZonesCreateCall interface {
		// Context(ctx context.Context) *ManagedZonesCreateCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (ManagedZone, error)
		// Fields(s ...googleapi.Field) *ManagedZonesCreateCall // TODO: Add as needed
	}

	ManagedZonesDeleteCall interface {
		// Context(ctx context.Context) *ManagedZonesDeleteCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) error
		// Fields(s ...googleapi.Field) *ManagedZonesDeleteCall // TODO: Add as needed
	}

	ManagedZonesGetCall interface {
		// Context(ctx context.Context) *ManagedZonesGetCall // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (ManagedZone, error)
		// Fields(s ...googleapi.Field) *ManagedZonesGetCall // TODO: Add as needed
		// IfNoneMatch(entityTag string) *ManagedZonesGetCall // TODO: Add as needed
	}

	ManagedZonesListCall interface {
		// Context(ctx context.Context) *ManagedZonesListCall // TODO: Add as needed
		DnsName(dnsName string) ManagedZonesListCall
		Do(opts ...googleapi.CallOption) (ManagedZonesListResponse, error)
		// Fields(s ...googleapi.Field) *ManagedZonesListCall // TODO: Add as needed
		// IfNoneMatch(entityTag string) *ManagedZonesListCall // TODO: Add as needed
		// MaxResults(maxResults int64) *ManagedZonesListCall // TODO: Add as needed
		// PageToken(pageToken string) *ManagedZonesListCall // TODO: Add as needed
		// Pages(ctx context.Context, f func(*ManagedZonesListResponse) error) error // TODO: Add as needed
	}

	ManagedZonesListResponse interface {
		// Kind() string // TODO: Add as needed
		// ManagedZones() []*ManagedZone // TODO: Add as needed
		ManagedZones() []ManagedZone
		// NextPageToken string // TODO: Add as needed
		// ServerResponse() googleapi.ServerResponse // TODO: Add as needed
		// ForceSendFields() []string // TODO: Add as needed
	}

	ManagedZonesService interface {
		// NewManagedZonesService(s *Service) *ManagedZonesService // TODO: Add to service if needed
		Create(project string, managedZone ManagedZone) ManagedZonesCreateCall
		Delete(project string, managedZone string) ManagedZonesDeleteCall
		Get(project string, managedZone string) ManagedZonesGetCall
		List(project string) ManagedZonesListCall
		NewManagedZone(dnsName string) ManagedZone
	}

	Project interface {
		// Id()  string  // TODO: Add as needed
		// Kind() string // TODO: Add as needed
		// Number() uint64 // TODO: Add as needed
		// Quota() *Quota // TODO: Add as needed
		// ServerResponse()  googleapi.ServerResponse // TODO: Add as needed
		// ForceSendFields() []string // TODO: Add as needed
	}

	ProjectsGetCall interface {
		// TODO: Add as needed
	}

	ProjectsService interface {
		// TODO: Add as needed
	}

	Quota interface {
		// TODO: Add as needed
	}

	ResourceRecordSet interface {
		// Kind() string // TODO: Add as needed
		Name() string
		Rrdatas() []string
		Ttl() int64
		Type() string
		// ForceSendFields []string  // TODO: Add as needed
	}

	ResourceRecordSetsListCall interface {
		// Context(ctx context.Context) *ResourceRecordSetsListCall  // TODO: Add as needed
		// Do(opts ...googleapi.CallOption) (*ResourceRecordSetsListResponse, error)  // TODO: Add as needed
		Do(opts ...googleapi.CallOption) (ResourceRecordSetsListResponse, error)
		// Fields(s ...googleapi.Field) *ResourceRecordSetsListCall  // TODO: Add as needed
		// IfNoneMatch(entityTag string) *ResourceRecordSetsListCall  // TODO: Add as needed
		// MaxResults(maxResults int64) *ResourceRecordSetsListCall  // TODO: Add as needed
		Name(name string) ResourceRecordSetsListCall
		// PageToken(pageToken string) *ResourceRecordSetsListCall  // TODO: Add as needed
		Type(type_ string) ResourceRecordSetsListCall
	}

	ResourceRecordSetsListResponse interface {
		// Kind() string  // TODO: Add as needed
		// NextPageToken() string  // TODO: Add as needed
		Rrsets() []ResourceRecordSet
		// ServerResponse() googleapi.ServerResponse  // TODO: Add as needed
		// ForceSendFields() []string  // TODO: Add as needed
	}

	ResourceRecordSetsService interface {
		// NewResourceRecordSetsService(s *Service) *ResourceRecordSetsService // TODO: add to service as needed
		List(project string, managedZone string) ResourceRecordSetsListCall
		NewResourceRecordSet(name string, rrdatas []string, ttl int64, type_ rrstype.RrsType) ResourceRecordSet
	}

	Service interface {
		// BasePath() string  // TODO: Add as needed
		// UserAgent() string // TODO: Add as needed
		Changes() ChangesService
		ManagedZones() ManagedZonesService
		Projects() ProjectsService
		ResourceRecordSets() ResourceRecordSetsService
	}
	// New(client *http.Client) (*Service, error)  // TODO: Add as needed
)
