/*
Copyright 2017 The Kubernetes Authors.

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

package aggregator

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/handler"
)

// SpecAggregator calls out to http handlers of APIServices and merges specs. It keeps state of the last
// known specs including the http etag.
type SpecAggregator interface {
	// AddUpdateService adds the services to the given spec name.
	// If the name is not registered, create a new nil spec under that name with the given services.
	AddUpdateService(name string, services ...*apiregistration.APIService) error
	// RemoveService removes the services with the given GroupVersions (they 1:1 relate) from the
	// spec with the given name. If there is no service left for that spec, it is removed.
	RemoveService(name string, services ...schema.GroupVersion) error

	// UpdateSpec updates the spec under the given name with the given etag.
	UpdateSpec(name string, spec *spec.Swagger, etag string) error
	// Spec returns the spec and etag with the given name.
	Spec(name string) (spec *spec.Swagger, etag string, exists bool)
}

type specAggregator struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// specs shared by APIServices indexed by a unique name
	specs map[string]*specInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *handler.OpenAPIService
}

var _ SpecAggregator = &specAggregator{}

// specInfo is used to store OpenAPI specs, its etag and the corresponding APIServices specified.
type specInfo struct {
	// those APIServices which share the spec. An APIService can only be part of one specInfo.
	apiServices []*apiregistration.APIService

	// specification of these API services. If null then the spec is not loaded yet.
	spec *spec.Swagger
	etag string
}

// NewSpecAggregator creates a new spec aggregator.
func NewSpecAggregator(openAPIVersionedService *handler.OpenAPIService) SpecAggregator {
	return &specAggregator{
		specs:                   map[string]*specInfo{},
		openAPIVersionedService: openAPIVersionedService,
	}
}

// buildOpenAPISpec aggregates the given specs.
func buildOpenAPISpec(specs map[string]*specInfo) (specToReturn *spec.Swagger, err error) {
	nonNilSpecs := []specInfo{}
	for _, specInfo := range specs {
		if specInfo.spec == nil {
			continue
		}
		nonNilSpecs = append(nonNilSpecs, *specInfo)
	}
	if len(nonNilSpecs) == 0 {
		return &spec.Swagger{}, nil
	}
	sortByPriority(nonNilSpecs)
	for _, si := range nonNilSpecs {
		if specToReturn == nil {
			specToReturn = &spec.Swagger{}
			*specToReturn = *si.spec
			// Paths and Definitions are set by MergeSpecsIgnorePathConflict
			specToReturn.Paths = nil
			specToReturn.Definitions = nil
		}
		if err := aggregator.MergeSpecsIgnorePathConflict(specToReturn, si.spec); err != nil {
			return nil, err
		}
	}
	return specToReturn, nil
}

// tryMergeSpecs tries to merge the new specs. If this succeeds, the newSpecInfos is written to
// s.specs and the OpenAPI service is informed about the update.
func (s *specAggregator) tryMergeSpecs(newSpecInfos map[string]*specInfo) error {
	mergedSpec, err := buildOpenAPISpec(newSpecInfos)
	if err != nil {
		return err
	}

	s.specs = newSpecInfos
	return s.openAPIVersionedService.UpdateSpec(mergedSpec)
}

// UpdateSpec updates the OpenAPI spec for the given name. It is thread safe.
func (s *specAggregator) UpdateSpec(name string, spec *spec.Swagger, etag string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	si, found := s.specs[name]
	if !found {
		return fmt.Errorf("OpenAPI spec for %q does not exists", name)
	}

	if si.etag == etag {
		return nil
	}

	// For APIServices (non-local) specs, only merge their /apis/ prefixed endpoint as it is the only paths
	// proxy handler delegates.
	spec = aggregator.FilterSpecByPathsWithoutSideEffects(spec, []string{"/apis/"})
	// TODO: use similar filtering to split the spec per APIService

	newSpecInfos := deepCopySpecs(s.specs)
	newSpecInfos[name].spec = spec
	newSpecInfos[name].etag = etag

	return s.tryMergeSpecs(newSpecInfos)
}

// AddUpdateSpec adds or updates the APIService to belong to the given name. If it was assigned
// to another name before, it is removed from there, potentially removing the whole spec.
// It is thread safe.
func (s *specAggregator) AddUpdateService(name string, services ...*apiregistration.APIService) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	// exit early
	if len(services) == 0 {
		return nil
	}

	gvs := make([]schema.GroupVersion, 0, len(services))
	for _, s := range services {
		gvs = append(gvs, schema.GroupVersion{s.Spec.Group, s.Spec.Version})
	}

	// remove services from other specs
	anyChange := false
	newSpecInfos := deepCopySpecs(s.specs)
	for n, si := range newSpecInfos {
		var changed bool
		si.apiServices, changed = filterOutServices(si.apiServices, gvs...)
		anyChange = anyChange || changed
		if n != name && len(si.apiServices) == 0 {
			delete(newSpecInfos, n)
		}
	}

	// add services to new name
	if _, found := newSpecInfos[name]; !found {
		newSpecInfos[name] = &specInfo{}
		anyChange = true
	}
	newSpecInfos[name].apiServices = append(newSpecInfos[name].apiServices, services...)

	// anything changed? This way we can also add nil specs without anything happening
	anyChange = anyChange || !deepEqualServices(s.specs[name].apiServices, newSpecInfos[name].apiServices)
	if !anyChange {
		return nil
	}

	return s.tryMergeSpecs(newSpecInfos)
}

// RemoveAPIServiceSpec removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveService(name string, services ...schema.GroupVersion) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	spec, found := s.specs[name]
	if !found {
		return nil
	}

	newServices, changed := filterOutServices(spec.apiServices, services...)
	if !changed {
		return nil
	}

	newSpecInfos := deepCopySpecs(s.specs)
	newSpecInfos[name].apiServices = newServices

	return s.tryMergeSpecs(newSpecInfos)
}

// Spec returns last known spec and etag.
func (s *specAggregator) Spec(name string) (spec *spec.Swagger, etag string, exists bool) {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	if spec, found := s.specs[name]; found {
		return spec.spec, spec.etag, true
	}
	return nil, "", false
}

func filterOutServices(services []*apiregistration.APIService, gvs ...schema.GroupVersion) ([]*apiregistration.APIService, bool) {
	newServices := make([]*apiregistration.APIService, 0, len(services))
	changed := false
nextService:
	for _, s := range services {
		for _, gv := range gvs {
			if s.Spec.Group == gv.Group && s.Spec.Version == gv.Version {
				changed = true
				continue nextService
			}
		}
		newServices = append(newServices, s)
	}
	if !changed {
		return services, false
	}
	return newServices, true
}

func deepCopySpecs(specs map[string]*specInfo) map[string]*specInfo {
	if specs == nil {
		return nil
	}

	clone := make(map[string]*specInfo, len(specs))
	for name, si := range specs {
		csi := *si
		csi.apiServices = append([]*apiregistration.APIService(nil), si.apiServices...)
		clone[name] = &csi
	}

	return clone
}

func deepEqualServices(a, b []*apiregistration.APIService) bool {
	as := map[string]*apiregistration.APIService{}
	bs := map[string]*apiregistration.APIService{}
	for _, x := range a {
		as[x.Name] = x
	}
	for _, x := range b {
		as[x.Name] = x
	}
	return reflect.DeepEqual(as, bs)
}
