/*
Copyright 2014 The Kubernetes Authors.

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

package resource

import (
	"fmt"
	"reflect"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
)

// ErrMatchFunc can be used to filter errors that may not be true failures.
type ErrMatchFunc func(error) bool

// Result contains helper methods for dealing with the outcome of a Builder.
type Result struct {
	err     error
	visitor Visitor

	sources            []Visitor
	singleItemImplied  bool
	targetsSingleItems bool

	mapper       *mapper
	ignoreErrors []utilerrors.Matcher

	// populated by a call to Infos
	info []*Info
}

// withError allows a fluent style for internal result code.
func (r *Result) withError(err error) *Result {
	r.err = err
	return r
}

// TargetsSingleItems returns true if any of the builder arguments pointed
// to non-list calls (if the user explicitly asked for any object by name).
// This includes directories, streams, URLs, and resource name tuples.
func (r *Result) TargetsSingleItems() bool {
	return r.targetsSingleItems
}

// IgnoreErrors will filter errors that occur when by visiting the result
// (but not errors that occur by creating the result in the first place),
// eliminating any that match fns. This is best used in combination with
// Builder.ContinueOnError(), where the visitors accumulate errors and return
// them after visiting as a slice of errors. If no errors remain after
// filtering, the various visitor methods on Result will return nil for
// err.
func (r *Result) IgnoreErrors(fns ...ErrMatchFunc) *Result {
	for _, fn := range fns {
		r.ignoreErrors = append(r.ignoreErrors, utilerrors.Matcher(fn))
	}
	return r
}

// Mapper returns a copy of the builder's mapper.
func (r *Result) Mapper() *mapper {
	return r.mapper
}

// Err returns one or more errors (via a util.ErrorList) that occurred prior
// to visiting the elements in the visitor. To see all errors including those
// that occur during visitation, invoke Infos().
func (r *Result) Err() error {
	return r.err
}

// Visit implements the Visitor interface on the items described in the Builder.
// Note that some visitor sources are not traversable more than once, or may
// return different results.  If you wish to operate on the same set of resources
// multiple times, use the Infos() method.
func (r *Result) Visit(fn VisitorFunc) error {
	if r.err != nil {
		return r.err
	}
	err := r.visitor.Visit(fn)
	return utilerrors.FilterOut(err, r.ignoreErrors...)
}

// IntoSingleItemImplied sets the provided boolean pointer to true if the Builder input
// implies a single item, or multiple.
func (r *Result) IntoSingleItemImplied(b *bool) *Result {
	*b = r.singleItemImplied
	return r
}

// Infos returns an array of all of the resource infos retrieved via traversal.
// Will attempt to traverse the entire set of visitors only once, and will return
// a cached list on subsequent calls.
func (r *Result) Infos() ([]*Info, error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.info != nil {
		return r.info, nil
	}

	infos := []*Info{}
	err := r.visitor.Visit(func(info *Info, err error) error {
		if err != nil {
			return err
		}
		infos = append(infos, info)
		return nil
	})
	err = utilerrors.FilterOut(err, r.ignoreErrors...)

	r.info, r.err = infos, err
	return infos, err
}

// Object returns a single object representing the output of a single visit to all
// found resources.  If the Builder was a singular context (expected to return a
// single resource by user input) and only a single resource was found, the resource
// will be returned as is.  Otherwise, the returned resources will be part of an
// v1.List. The ResourceVersion of the v1.List will be set only if it is identical
// across all infos returned.
func (r *Result) Object() (runtime.Object, error) {
	infos, err := r.Infos()
	if err != nil {
		return nil, err
	}

	versions := sets.String{}
	objects := []runtime.Object{}
	for _, info := range infos {
		if info.Object != nil {
			objects = append(objects, info.Object)
			versions.Insert(info.ResourceVersion)
		}
	}

	if len(objects) == 1 {
		if r.singleItemImplied {
			return objects[0], nil
		}
		// if the item is a list already, don't create another list
		if meta.IsListType(objects[0]) {
			return objects[0], nil
		}
	}

	version := ""
	if len(versions) == 1 {
		version = versions.List()[0]
	}

	return toV1List(objects, version), err
}

// Compile time check to enforce that list implements the necessary interface
var _ metav1.ListInterface = &v1.List{}
var _ metav1.ListMetaAccessor = &v1.List{}

// toV1List takes a slice of Objects + their version, and returns
// a v1.List Object containing the objects in the Items field
func toV1List(objects []runtime.Object, version string) runtime.Object {
	raw := []runtime.RawExtension{}
	for _, o := range objects {
		raw = append(raw, runtime.RawExtension{Object: o})
	}
	return &v1.List{
		ListMeta: metav1.ListMeta{
			ResourceVersion: version,
		},
		Items: raw,
	}
}

// ResourceMapping returns a single meta.RESTMapping representing the
// resources located by the builder, or an error if more than one
// mapping was found.
func (r *Result) ResourceMapping() (*meta.RESTMapping, error) {
	if r.err != nil {
		return nil, r.err
	}
	mappings := map[schema.GroupVersionResource]*meta.RESTMapping{}
	for i := range r.sources {
		m, ok := r.sources[i].(ResourceMapping)
		if !ok {
			return nil, fmt.Errorf("a resource mapping could not be loaded from %v", reflect.TypeOf(r.sources[i]))
		}
		mapping := m.ResourceMapping()
		mappings[mapping.Resource] = mapping
	}
	if len(mappings) != 1 {
		return nil, fmt.Errorf("expected only a single resource type")
	}
	for _, mapping := range mappings {
		return mapping, nil
	}
	return nil, nil
}

// Watch retrieves changes that occur on the server to the specified resource.
// It currently supports watching a single source - if the resource source
// (selectors or pure types) can be watched, they will be, otherwise the list
// will be visited (equivalent to the Infos() call) and if there is a single
// resource present, it will be watched, otherwise an error will be returned.
func (r *Result) Watch(resourceVersion string) (watch.Interface, error) {
	if r.err != nil {
		return nil, r.err
	}
	if len(r.sources) != 1 {
		return nil, fmt.Errorf("you may only watch a single resource or type of resource at a time")
	}
	w, ok := r.sources[0].(Watchable)
	if !ok {
		info, err := r.Infos()
		if err != nil {
			return nil, err
		}
		if len(info) != 1 {
			return nil, fmt.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(info))
		}
		return info[0].Watch(resourceVersion)
	}
	return w.Watch(resourceVersion)
}
