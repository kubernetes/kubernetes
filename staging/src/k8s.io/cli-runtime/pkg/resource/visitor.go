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
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
)

const (
	constSTDINstr       = "STDIN"
	stopValidateMessage = "if you choose to ignore these errors, turn validation off with --validate=false"
)

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
	// Client will only be present if this builder was not local
	Client RESTClient
	// Mapping will only be present if this builder was not local
	Mapping *meta.RESTMapping

	// Namespace will be set if the object is namespaced and has a specified value.
	Namespace string
	Name      string

	// Optional, Source is the filename or URL to template file (.json or .yaml),
	// or stdin to use to handle the resource
	Source string
	// Optional, this is the most recent value returned by the server if available. It will
	// typically be in unstructured or internal forms, depending on how the Builder was
	// defined. If retrieved from the server, the Builder expects the mapping client to
	// decide the final form. Use the AsVersioned, AsUnstructured, and AsInternal helpers
	// to alter the object versions.
	// If Subresource is specified, this will be the object for the subresource.
	Object runtime.Object
	// Optional, this is the most recent resource version the server knows about for
	// this type of resource. It may not match the resource version of the object,
	// but if set it should be equal to or newer than the resource version of the
	// object (however the server defines resource version).
	ResourceVersion string
	// Optional, if specified, the object is the most recent value of the subresource
	// returned by the server if available.
	Subresource string
}

// Visit implements Visitor
func (i *Info) Visit(fn VisitorFunc) error {
	return fn(i, nil)
}

// Get retrieves the object from the Namespace and Name fields
func (i *Info) Get() (err error) {
	obj, err := NewHelper(i.Client, i.Mapping).WithSubresource(i.Subresource).Get(i.Namespace, i.Name)
	if err != nil {
		if errors.IsNotFound(err) && len(i.Namespace) > 0 && i.Namespace != metav1.NamespaceDefault && i.Namespace != metav1.NamespaceAll {
			err2 := i.Client.Get().AbsPath("api", "v1", "namespaces", i.Namespace).Do(context.TODO()).Error()
			if err2 != nil && errors.IsNotFound(err2) {
				return err2
			}
		}
		return err
	}
	i.Object = obj
	i.ResourceVersion, _ = metadataAccessor.ResourceVersion(obj)
	return nil
}

// Refresh updates the object with another object. If ignoreError is set
// the Object will be updated even if name, namespace, or resourceVersion
// attributes cannot be loaded from the object.
func (i *Info) Refresh(obj runtime.Object, ignoreError bool) error {
	name, err := metadataAccessor.Name(obj)
	if err != nil {
		if !ignoreError {
			return err
		}
	} else {
		i.Name = name
	}
	namespace, err := metadataAccessor.Namespace(obj)
	if err != nil {
		if !ignoreError {
			return err
		}
	} else {
		i.Namespace = namespace
	}
	version, err := metadataAccessor.ResourceVersion(obj)
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

// ObjectName returns an approximate form of the resource's kind/name.
func (i *Info) ObjectName() string {
	if i.Mapping != nil {
		return fmt.Sprintf("%s/%s", i.Mapping.Resource.Resource, i.Name)
	}
	gvk := i.Object.GetObjectKind().GroupVersionKind()
	if len(gvk.Group) == 0 {
		return fmt.Sprintf("%s/%s", strings.ToLower(gvk.Kind), i.Name)
	}
	return fmt.Sprintf("%s.%s/%s\n", strings.ToLower(gvk.Kind), gvk.Group, i.Name)
}

// String returns the general purpose string representation
func (i *Info) String() string {
	basicInfo := fmt.Sprintf("Name: %q, Namespace: %q", i.Name, i.Namespace)
	if i.Mapping != nil {
		mappingInfo := fmt.Sprintf("Resource: %q, GroupVersionKind: %q", i.Mapping.Resource.String(),
			i.Mapping.GroupVersionKind.String())
		return fmt.Sprint(mappingInfo, "\n", basicInfo)
	}
	return basicInfo
}

// Namespaced returns true if the object belongs to a namespace
func (i *Info) Namespaced() bool {
	if i.Mapping != nil {
		// if we have RESTMapper info, use it
		return i.Mapping.Scope.Name() == meta.RESTScopeNameNamespace
	}
	// otherwise, use the presence of a namespace in the info as an indicator
	return len(i.Namespace) > 0
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

type ConcurrentVisitorList struct {
	visitors    []Visitor
	concurrency int
}

func (l ConcurrentVisitorList) Visit(fn VisitorFunc) error {
	g := errgroup.Group{}

	// Concurrency 1 just runs the visitors sequentially, this is the default
	// as it preserves the previous behavior, but allows components to opt into
	// concurrency.
	concurrency := 1
	if l.concurrency > concurrency {
		concurrency = l.concurrency
	}
	g.SetLimit(concurrency)

	for i := range l.visitors {
		i := i
		g.Go(func() error {
			return l.visitors[i].Visit(fn)
		})
	}

	return g.Wait()
}

// EagerVisitorList implements Visit for the sub visitors it contains. All errors
// will be captured and returned at the end of iteration.
type EagerVisitorList []Visitor

// Visit implements Visitor, and gathers errors that occur during processing until
// all sub visitors have been visited.
func (l EagerVisitorList) Visit(fn VisitorFunc) error {
	var errs []error
	for i := range l {
		err := l[i].Visit(func(info *Info, err error) error {
			if err != nil {
				errs = append(errs, err)
				return nil
			}
			if err := fn(info, nil); err != nil {
				errs = append(errs, err)
			}
			return nil
		})
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

func ValidateSchema(data []byte, schema ContentValidator) error {
	if schema == nil {
		return nil
	}
	if err := schema.ValidateBytes(data); err != nil {
		return fmt.Errorf("error validating data: %v; %s", err, stopValidateMessage)
	}
	return nil
}

// URLVisitor downloads the contents of a URL, and if successful, returns
// an info object representing the downloaded object.
type URLVisitor struct {
	URL *url.URL
	*StreamVisitor
	HttpAttemptCount int
}

func (v *URLVisitor) Visit(fn VisitorFunc) error {
	body, err := readHttpWithRetries(httpgetImpl, time.Second, v.URL.String(), v.HttpAttemptCount)
	if err != nil {
		return err
	}
	defer body.Close()
	v.StreamVisitor.Reader = body
	return v.StreamVisitor.Visit(fn)
}

// readHttpWithRetries tries to http.Get the v.URL retries times before giving up.
func readHttpWithRetries(get httpget, duration time.Duration, u string, attempts int) (io.ReadCloser, error) {
	var err error
	if attempts <= 0 {
		return nil, fmt.Errorf("http attempts must be greater than 0, was %d", attempts)
	}
	for i := 0; i < attempts; i++ {
		var (
			statusCode int
			status     string
			body       io.ReadCloser
		)
		if i > 0 {
			time.Sleep(duration)
		}

		// Try to get the URL
		statusCode, status, body, err = get(u)

		// Retry Errors
		if err != nil {
			continue
		}

		if statusCode == http.StatusOK {
			return body, nil
		}
		body.Close()
		// Error - Set the error condition from the StatusCode
		err = fmt.Errorf("unable to read URL %q, server reported %s, status code=%d", u, status, statusCode)

		if statusCode >= 500 && statusCode < 600 {
			// Retry 500's
			continue
		} else {
			// Don't retry other StatusCodes
			break
		}
	}
	return nil, err
}

// httpget Defines function to retrieve a url and return the results.  Exists for unit test stubbing.
type httpget func(url string) (int, string, io.ReadCloser, error)

// httpgetImpl Implements a function to retrieve a url and return the results.
func httpgetImpl(url string) (int, string, io.ReadCloser, error) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, "", nil, err
	}
	return resp.StatusCode, resp.Status, resp.Body, nil
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
	return v.visitor.Visit(func(info *Info, err error) error {
		if err != nil {
			return err
		}
		for i := range v.decorators {
			if err := v.decorators[i](info, nil); err != nil {
				return err
			}
		}
		return fn(info, nil)
	})
}

// ContinueOnErrorVisitor visits each item and, if an error occurs on
// any individual item, returns an aggregate error after all items
// are visited.
type ContinueOnErrorVisitor struct {
	Visitor
}

// Visit returns nil if no error occurs during traversal, a regular
// error if one occurs, or if multiple errors occur, an aggregate
// error.  If the provided visitor fails on any individual item it
// will not prevent the remaining items from being visited. An error
// returned by the visitor directly may still result in some items
// not being visited.
func (v ContinueOnErrorVisitor) Visit(fn VisitorFunc) error {
	var errs []error
	err := v.Visitor.Visit(func(info *Info, err error) error {
		if err != nil {
			errs = append(errs, err)
			return nil
		}
		if err := fn(info, nil); err != nil {
			errs = append(errs, err)
		}
		return nil
	})
	if err != nil {
		errs = append(errs, err)
	}
	if len(errs) == 1 {
		return errs[0]
	}
	return utilerrors.NewAggregate(errs)
}

// FlattenListVisitor flattens any objects that runtime.ExtractList recognizes as a list
// - has an "Items" public field that is a slice of runtime.Objects or objects satisfying
// that interface - into multiple Infos. Returns nil in the case of no errors.
// When an error is hit on sub items (for instance, if a List contains an object that does
// not have a registered client or resource), returns an aggregate error.
type FlattenListVisitor struct {
	visitor Visitor
	typer   runtime.ObjectTyper
	mapper  *mapper
}

// NewFlattenListVisitor creates a visitor that will expand list style runtime.Objects
// into individual items and then visit them individually.
func NewFlattenListVisitor(v Visitor, typer runtime.ObjectTyper, mapper *mapper) Visitor {
	return FlattenListVisitor{v, typer, mapper}
}

func (v FlattenListVisitor) Visit(fn VisitorFunc) error {
	return v.visitor.Visit(func(info *Info, err error) error {
		if err != nil {
			return err
		}
		if info.Object == nil {
			return fn(info, nil)
		}
		if !meta.IsListType(info.Object) {
			return fn(info, nil)
		}

		items := []runtime.Object{}
		itemsToProcess := []runtime.Object{info.Object}

		for i := 0; i < len(itemsToProcess); i++ {
			currObj := itemsToProcess[i]
			if !meta.IsListType(currObj) {
				items = append(items, currObj)
				continue
			}

			currItems, err := meta.ExtractList(currObj)
			if err != nil {
				return err
			}
			if errs := runtime.DecodeList(currItems, v.mapper.decoder); len(errs) > 0 {
				return utilerrors.NewAggregate(errs)
			}
			itemsToProcess = append(itemsToProcess, currItems...)
		}

		// If we have a GroupVersionKind on the list, prioritize that when asking for info on the objects contained in the list
		var preferredGVKs []schema.GroupVersionKind
		if info.Mapping != nil && !info.Mapping.GroupVersionKind.Empty() {
			preferredGVKs = append(preferredGVKs, info.Mapping.GroupVersionKind)
		}
		var errs []error
		for i := range items {
			item, err := v.mapper.infoForObject(items[i], v.typer, preferredGVKs)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			if len(info.ResourceVersion) != 0 {
				item.ResourceVersion = info.ResourceVersion
			}
			// propagate list source to items source
			if len(info.Source) != 0 {
				item.Source = info.Source
			}
			if err := fn(item, nil); err != nil {
				errs = append(errs, err)
			}
		}
		return utilerrors.NewAggregate(errs)
	})
}

func ignoreFile(path string, extensions []string) bool {
	if len(extensions) == 0 {
		return false
	}
	ext := filepath.Ext(path)
	for _, s := range extensions {
		if s == ext {
			return false
		}
	}
	return true
}

// FileVisitorForSTDIN return a special FileVisitor just for STDIN
func FileVisitorForSTDIN(mapper *mapper, schema ContentValidator) Visitor {
	return &FileVisitor{
		Path:          constSTDINstr,
		StreamVisitor: NewStreamVisitor(nil, mapper, constSTDINstr, schema),
	}
}

// ExpandPathsToFileVisitors will return a slice of FileVisitors that will handle files from the provided path.
// After FileVisitors open the files, they will pass an io.Reader to a StreamVisitor to do the reading. (stdin
// is also taken care of). Paths argument also accepts a single file, and will return a single visitor
func ExpandPathsToFileVisitors(mapper *mapper, paths string, recursive bool, extensions []string, schema ContentValidator) ([]Visitor, error) {
	var visitors []Visitor
	err := filepath.Walk(paths, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if fi.IsDir() {
			if path != paths && !recursive {
				return filepath.SkipDir
			}
			return nil
		}
		// Don't check extension if the filepath was passed explicitly
		if path != paths && ignoreFile(path, extensions) {
			return nil
		}

		visitor := &FileVisitor{
			Path:          path,
			StreamVisitor: NewStreamVisitor(nil, mapper, path, schema),
		}

		visitors = append(visitors, visitor)
		return nil
	})

	if err != nil {
		return nil, err
	}
	return visitors, nil
}

// FileVisitor is wrapping around a StreamVisitor, to handle open/close files
type FileVisitor struct {
	Path string
	*StreamVisitor
}

// Visit in a FileVisitor is just taking care of opening/closing files
func (v *FileVisitor) Visit(fn VisitorFunc) error {
	var f *os.File
	if v.Path == constSTDINstr {
		f = os.Stdin
	} else {
		var err error
		f, err = os.Open(v.Path)
		if err != nil {
			return err
		}
		defer f.Close()
	}

	// TODO: Consider adding a flag to force to UTF16, apparently some
	// Windows tools don't write the BOM
	utf16bom := unicode.BOMOverride(unicode.UTF8.NewDecoder())
	v.StreamVisitor.Reader = transform.NewReader(f, utf16bom)

	return v.StreamVisitor.Visit(fn)
}

// StreamVisitor reads objects from an io.Reader and walks them. A stream visitor can only be
// visited once.
// TODO: depends on objects being in JSON format before being passed to decode - need to implement
// a stream decoder method on runtime.Codec to properly handle this.
type StreamVisitor struct {
	io.Reader
	*mapper

	Source string
	Schema ContentValidator
}

// NewStreamVisitor is a helper function that is useful when we want to change the fields of the struct but keep calls the same.
func NewStreamVisitor(r io.Reader, mapper *mapper, source string, schema ContentValidator) *StreamVisitor {
	return &StreamVisitor{
		Reader: r,
		mapper: mapper,
		Source: source,
		Schema: schema,
	}
}

// Visit implements Visitor over a stream. StreamVisitor is able to distinct multiple resources in one stream.
func (v *StreamVisitor) Visit(fn VisitorFunc) error {
	d := yaml.NewYAMLOrJSONDecoder(v.Reader, 4096)
	for {
		ext := runtime.RawExtension{}
		if err := d.Decode(&ext); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("error parsing %s: %v", v.Source, err)
		}
		// TODO: This needs to be able to handle object in other encodings and schemas.
		ext.Raw = bytes.TrimSpace(ext.Raw)
		if len(ext.Raw) == 0 || bytes.Equal(ext.Raw, []byte("null")) {
			continue
		}
		if err := ValidateSchema(ext.Raw, v.Schema); err != nil {
			return fmt.Errorf("error validating %q: %v", v.Source, err)
		}
		info, err := v.infoForData(ext.Raw, v.Source)
		if err != nil {
			if fnErr := fn(info, err); fnErr != nil {
				return fnErr
			}
			continue
		}
		if err := fn(info, nil); err != nil {
			return err
		}
	}
}

func UpdateObjectNamespace(info *Info, err error) error {
	if err != nil {
		return err
	}
	if info.Object != nil {
		return metadataAccessor.SetNamespace(info.Object, info.Namespace)
	}
	return nil
}

// FilterNamespace omits the namespace if the object is not namespace scoped
func FilterNamespace(info *Info, err error) error {
	if err != nil {
		return err
	}
	if !info.Namespaced() {
		info.Namespace = ""
		UpdateObjectNamespace(info, nil)
	}
	return nil
}

// SetNamespace ensures that every Info object visited will have a namespace
// set. If info.Object is set, it will be mutated as well.
func SetNamespace(namespace string) VisitorFunc {
	return func(info *Info, err error) error {
		if err != nil {
			return err
		}
		if !info.Namespaced() {
			return nil
		}
		if len(info.Namespace) == 0 {
			info.Namespace = namespace
			UpdateObjectNamespace(info, nil)
		}
		return nil
	}
}

// RequireNamespace will either set a namespace if none is provided on the
// Info object, or if the namespace is set and does not match the provided
// value, returns an error. This is intended to guard against administrators
// accidentally operating on resources outside their namespace.
func RequireNamespace(namespace string) VisitorFunc {
	return func(info *Info, err error) error {
		if err != nil {
			return err
		}
		if !info.Namespaced() {
			return nil
		}
		if len(info.Namespace) == 0 {
			info.Namespace = namespace
			UpdateObjectNamespace(info, nil)
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
func RetrieveLatest(info *Info, err error) error {
	if err != nil {
		return err
	}
	if meta.IsListType(info.Object) {
		return fmt.Errorf("watch is only supported on individual resources and resource collections, but a list of resources is found")
	}
	if len(info.Name) == 0 {
		return nil
	}
	if info.Namespaced() && len(info.Namespace) == 0 {
		return fmt.Errorf("no namespace set on resource %s %q", info.Mapping.Resource, info.Name)
	}
	return info.Get()
}

// RetrieveLazy updates the object if it has not been loaded yet.
func RetrieveLazy(info *Info, err error) error {
	if err != nil {
		return err
	}
	if info.Object == nil {
		return info.Get()
	}
	return nil
}

type FilterFunc func(info *Info, err error) (bool, error)

type FilteredVisitor struct {
	visitor Visitor
	filters []FilterFunc
}

func NewFilteredVisitor(v Visitor, fn ...FilterFunc) Visitor {
	if len(fn) == 0 {
		return v
	}
	return FilteredVisitor{v, fn}
}

func (v FilteredVisitor) Visit(fn VisitorFunc) error {
	return v.visitor.Visit(func(info *Info, err error) error {
		if err != nil {
			return err
		}
		for _, filter := range v.filters {
			ok, err := filter(info, nil)
			if err != nil {
				return err
			}
			if !ok {
				return nil
			}
		}
		return fn(info, nil)
	})
}

func FilterByLabelSelector(s labels.Selector) FilterFunc {
	return func(info *Info, err error) (bool, error) {
		if err != nil {
			return false, err
		}
		a, err := meta.Accessor(info.Object)
		if err != nil {
			return false, err
		}
		if !s.Matches(labels.Set(a.GetLabels())) {
			return false, nil
		}
		return true, nil
	}
}

type InfoListVisitor []*Info

func (infos InfoListVisitor) Visit(fn VisitorFunc) error {
	var err error
	for _, i := range infos {
		err = fn(i, err)
	}
	return err
}
