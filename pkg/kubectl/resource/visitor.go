/*
Copyright 2014 Google Inc. All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"

	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Visitor lets clients walk a list of resources.
type Visitor interface {
	Visit(VisitorFunc) error
}

// VisitorFunc implements the Visitor interface for a matching function
type VisitorFunc func(*Info) error

// Watchable describes a resource that can be watched for changes that occur on the server,
// beginning after the provided resource version.
type Watchable interface {
	Watch(resourceVersion string) (watch.Interface, error)
}

// ResourceMapping allows an object to return the resource mapping associated with
// the resource or resources it represents.
type ResourceMapping interface {
	ResourceMapping() *meta.RESTMapping
}

// Info contains temporary info to execute a REST call, or show the results
// of an already completed REST call.
type Info struct {
	Client    RESTClient
	Mapping   *meta.RESTMapping
	Namespace string
	Name      string

	// Optional, this is the most recent value returned by the server if available
	runtime.Object
	// Optional, this is the most recent resource version the server knows about for
	// this type of resource. It may not match the resource version of the object,
	// but if set it should be equal to or newer than the resource version of the
	// object (however the server defines resource version).
	ResourceVersion string
}

// NewInfo returns a new info object
func NewInfo(client RESTClient, mapping *meta.RESTMapping, namespace, name string) *Info {
	return &Info{
		Client:    client,
		Mapping:   mapping,
		Namespace: namespace,
		Name:      name,
	}
}

// Visit implements Visitor
func (i *Info) Visit(fn VisitorFunc) error {
	return fn(i)
}

// Get retrieves the object from the Namespace and Name fields
func (i *Info) Get() error {
	obj, err := NewHelper(i.Client, i.Mapping).Get(i.Namespace, i.Name)
	if err != nil {
		return err
	}
	i.Object = obj
	i.ResourceVersion, _ = i.Mapping.MetadataAccessor.ResourceVersion(obj)
	return nil
}

// Refresh updates the object with another object. If ignoreError is set
// the Object will be updated even if name, namespace, or resourceVersion
// attributes cannot be loaded from the object.
func (i *Info) Refresh(obj runtime.Object, ignoreError bool) error {
	name, err := i.Mapping.MetadataAccessor.Name(obj)
	if err != nil {
		if !ignoreError {
			return err
		}
	} else {
		i.Name = name
	}
	namespace, err := i.Mapping.MetadataAccessor.Namespace(obj)
	if err != nil {
		if !ignoreError {
			return err
		}
	} else {
		i.Namespace = namespace
	}
	version, err := i.Mapping.MetadataAccessor.ResourceVersion(obj)
	if err != nil {
		if !ignoreError {
			return err
		}
	} else {
		i.ResourceVersion = version
	}
	i.Object = obj
	return nil
}

// Namespaced returns true if the object belongs to a namespace
func (i *Info) Namespaced() bool {
	return i.Mapping != nil && i.Mapping.Scope.Name() == meta.RESTScopeNameNamespace
}

// Watch returns server changes to this object after it was retrieved.
func (i *Info) Watch(resourceVersion string) (watch.Interface, error) {
	return NewHelper(i.Client, i.Mapping).WatchSingle(i.Namespace, i.Name, resourceVersion)
}

// ResourceMapping returns the mapping for this resource and implements ResourceMapping
func (i *Info) ResourceMapping() *meta.RESTMapping {
	return i.Mapping
}

// VisitorList implements Visit for the sub visitors it contains. The first error
// returned from a child Visitor will terminate iteration.
type VisitorList []Visitor

// Visit implements Visitor
func (l VisitorList) Visit(fn VisitorFunc) error {
	for i := range l {
		if err := l[i].Visit(fn); err != nil {
			return err
		}
	}
	return nil
}

// EagerVisitorList implements Visit for the sub visitors it contains. All errors
// will be captured and returned at the end of iteration.
type EagerVisitorList []Visitor

// Visit implements Visitor, and gathers errors that occur during processing until
// all sub visitors have been visited.
func (l EagerVisitorList) Visit(fn VisitorFunc) error {
	errs := []error(nil)
	for i := range l {
		if err := l[i].Visit(func(info *Info) error {
			if err := fn(info); err != nil {
				errs = append(errs, err)
			}
			return nil
		}); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.NewAggregate(errs)
}

// PathVisitor visits a given path and returns an object representing the file
// at that path.
type PathVisitor struct {
	*Mapper
	// The file path to load
	Path string
	// Whether to ignore files that are not recognized as API objects
	IgnoreErrors bool
}

func (v *PathVisitor) Visit(fn VisitorFunc) error {
	data, err := ioutil.ReadFile(v.Path)
	if err != nil {
		return fmt.Errorf("unable to read %q: %v", v.Path, err)
	}
	info, err := v.Mapper.InfoForData(data, v.Path)
	if err != nil {
		if v.IgnoreErrors {
			return err
		}
		glog.V(2).Infof("Unable to load file %q: %v", v.Path, err)
		return nil
	}
	return fn(info)
}

// DirectoryVisitor loads the specified files from a directory and passes them
// to visitors.
type DirectoryVisitor struct {
	*Mapper
	// The directory or file to start from
	Path string
	// Whether directories are recursed
	Recursive bool
	// The file extensions to include. If empty, all files are read.
	Extensions []string
	// Whether to ignore files that are not recognized as API objects
	IgnoreErrors bool
}

func (v *DirectoryVisitor) ignoreFile(path string) bool {
	if len(v.Extensions) == 0 {
		return false
	}
	ext := filepath.Ext(path)
	for _, s := range v.Extensions {
		if s == ext {
			return false
		}
	}
	return true
}

func (v *DirectoryVisitor) Visit(fn VisitorFunc) error {
	return filepath.Walk(v.Path, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if fi.IsDir() {
			if path != v.Path && !v.Recursive {
				return filepath.SkipDir
			}
			return nil
		}
		if v.ignoreFile(path) {
			return nil
		}

		data, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("unable to read %q: %v", path, err)
		}
		info, err := v.Mapper.InfoForData(data, path)
		if err != nil {
			if v.IgnoreErrors {
				return err
			}
			glog.V(2).Infof("Unable to load file %q: %v", path, err)
			return nil
		}
		return fn(info)
	})
}

// URLVisitor downloads the contents of a URL, and if successful, returns
// an info object representing the downloaded object.
type URLVisitor struct {
	*Mapper
	URL *url.URL
}

func (v *URLVisitor) Visit(fn VisitorFunc) error {
	res, err := http.Get(v.URL.String())
	if err != nil {
		return fmt.Errorf("unable to access URL %q: %v\n", v.URL, err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return fmt.Errorf("unable to read URL %q, server reported %d %s", v.URL, res.StatusCode, res.Status)
	}
	data, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("unable to read URL %q: %v\n", v.URL, err)
	}
	info, err := v.Mapper.InfoForData(data, v.URL.String())
	if err != nil {
		return err
	}
	return fn(info)
}

// DecoratedVisitor will invoke the decorators in order prior to invoking the visitor function
// passed to Visit. An error will terminate the visit.
type DecoratedVisitor struct {
	visitor    Visitor
	decorators []VisitorFunc
}

// NewDecoratedVisitor will create a visitor that invokes the provided visitor functions before
// the user supplied visitor function is invoked, giving them the opportunity to mutate the Info
// object or terminate early with an error.
func NewDecoratedVisitor(v Visitor, fn ...VisitorFunc) Visitor {
	if len(fn) == 0 {
		return v
	}
	return DecoratedVisitor{v, fn}
}

// Visit implements Visitor
func (v DecoratedVisitor) Visit(fn VisitorFunc) error {
	return v.visitor.Visit(func(info *Info) error {
		for i := range v.decorators {
			if err := v.decorators[i](info); err != nil {
				return err
			}
		}
		return fn(info)
	})
}

// FlattenListVisitor flattens any objects that runtime.ExtractList recognizes as a list
// - has an "Items" public field that is a slice of runtime.Objects or objects satisfying
// that interface - into multiple Infos. An error on any sub item (for instance, if a List
// contains an object that does not have a registered client or resource) will terminate
// the visit.
// TODO: allow errors to be aggregated?
type FlattenListVisitor struct {
	Visitor
	*Mapper
}

// NewFlattenListVisitor creates a visitor that will expand list style runtime.Objects
// into individual items and then visit them individually.
func NewFlattenListVisitor(v Visitor, mapper *Mapper) Visitor {
	return FlattenListVisitor{v, mapper}
}

func (v FlattenListVisitor) Visit(fn VisitorFunc) error {
	return v.Visitor.Visit(func(info *Info) error {
		if info.Object == nil {
			return fn(info)
		}
		items, err := runtime.ExtractList(info.Object)
		if err != nil {
			return fn(info)
		}
		if errs := runtime.DecodeList(items, struct {
			runtime.ObjectTyper
			runtime.Decoder
		}{v.Mapper, info.Mapping.Codec}); len(errs) > 0 {
			return errors.NewAggregate(errs)
		}
		for i := range items {
			item, err := v.InfoForObject(items[i])
			if err != nil {
				return err
			}
			if len(info.ResourceVersion) != 0 {
				item.ResourceVersion = info.ResourceVersion
			}
			if err := fn(item); err != nil {
				return err
			}
		}
		return nil
	})
}

// StreamVisitor reads objects from an io.Reader and walks them. A stream visitor can only be
// visited once.
// TODO: depends on objects being in JSON format before being passed to decode - need to implement
// a stream decoder method on runtime.Codec to properly handle this.
type StreamVisitor struct {
	io.Reader
	*Mapper

	Source       string
	IgnoreErrors bool
}

// NewStreamVisitor creates a visitor that will return resources that were encoded into the provided
// stream. If ignoreErrors is set, unrecognized or invalid objects will be skipped and logged. An
// empty stream is treated as an error for now.
// TODO: convert ignoreErrors into a func(data, error, count) bool that consumers can use to decide
// what to do with ignored errors.
func NewStreamVisitor(r io.Reader, mapper *Mapper, source string, ignoreErrors bool) Visitor {
	return &StreamVisitor{r, mapper, source, ignoreErrors}
}

// Visit implements Visitor over a stream.
func (v *StreamVisitor) Visit(fn VisitorFunc) error {
	d := yaml.NewYAMLOrJSONDecoder(v.Reader, 4096)
	for {
		ext := runtime.RawExtension{}
		if err := d.Decode(&ext); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		ext.RawJSON = bytes.TrimSpace(ext.RawJSON)
		if len(ext.RawJSON) == 0 || bytes.Equal(ext.RawJSON, []byte("null")) {
			continue
		}
		info, err := v.InfoForData(ext.RawJSON, v.Source)
		if err != nil {
			if v.IgnoreErrors {
				glog.Warningf("Could not read an encoded object from %s: %v", v.Source, err)
				glog.V(4).Infof("Unreadable: %s", string(ext.RawJSON))
				continue
			}
			return err
		}
		if err := fn(info); err != nil {
			return err
		}
	}
	return nil
}

func UpdateObjectNamespace(info *Info) error {
	if info.Object != nil {
		return info.Mapping.MetadataAccessor.SetNamespace(info.Object, info.Namespace)
	}
	return nil
}

// FilterNamespace omits the namespace if the object is not namespace scoped
func FilterNamespace(info *Info) error {
	if !info.Namespaced() {
		info.Namespace = ""
		UpdateObjectNamespace(info)
	}
	return nil
}

// SetNamespace ensures that every Info object visited will have a namespace
// set. If info.Object is set, it will be mutated as well.
func SetNamespace(namespace string) VisitorFunc {
	return func(info *Info) error {
		if len(info.Namespace) == 0 {
			info.Namespace = namespace
			UpdateObjectNamespace(info)
		}
		return nil
	}
}

// RequireNamespace will either set a namespace if none is provided on the
// Info object, or if the namespace is set and does not match the provided
// value, returns an error. This is intended to guard against administrators
// accidentally operating on resources outside their namespace.
func RequireNamespace(namespace string) VisitorFunc {
	return func(info *Info) error {
		if !info.Namespaced() {
			return nil
		}
		if len(info.Namespace) == 0 {
			info.Namespace = namespace
			UpdateObjectNamespace(info)
			return nil
		}
		if info.Namespace != namespace {
			return fmt.Errorf("the namespace from the provided object %q does not match the namespace %q. You must pass '--namespace=%s' to perform this operation.", info.Namespace, namespace, info.Namespace)
		}
		return nil
	}
}

// RetrieveLatest updates the Object on each Info by invoking a standard client
// Get.
func RetrieveLatest(info *Info) error {
	if len(info.Name) == 0 {
		return nil
	}
	if info.Namespaced() && len(info.Namespace) == 0 {
		return fmt.Errorf("no namespace set on resource %s %q", info.Mapping.Resource, info.Name)
	}
	obj, err := NewHelper(info.Client, info.Mapping).Get(info.Namespace, info.Name)
	if err != nil {
		return err
	}
	info.Object = obj
	info.ResourceVersion, _ = info.Mapping.MetadataAccessor.ResourceVersion(obj)
	return nil
}

// RetrieveLazy updates the object if it has not been loaded yet.
func RetrieveLazy(info *Info) error {
	if info.Object == nil {
		return info.Get()
	}
	return nil
}
