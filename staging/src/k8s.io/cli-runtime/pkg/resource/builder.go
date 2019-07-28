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
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/restmapper"
)

var FileExtensions = []string{".json", ".yaml", ".yml"}
var InputExtensions = append(FileExtensions, "stdin")

const defaultHttpGetAttempts int = 3

// Builder provides convenience functions for taking arguments and parameters
// from the command line and converting them to a list of resources to iterate
// over using the Visitor interface.
type Builder struct {
	categoryExpanderFn CategoryExpanderFunc

	// mapper is set explicitly by resource builders
	mapper *mapper

	// clientConfigFn is a function to produce a client, *if* you need one
	clientConfigFn ClientConfigFunc

	restMapperFn RESTMapperFunc

	// objectTyper is statically determinant per-command invocation based on your internal or unstructured choice
	// it does not ever need to rely upon discovery.
	objectTyper runtime.ObjectTyper

	// codecFactory describes which codecs you want to use
	negotiatedSerializer runtime.NegotiatedSerializer

	// local indicates that we cannot make server calls
	local bool

	errs []error

	paths  []Visitor
	stream bool
	dir    bool

	labelSelector     *string
	fieldSelector     *string
	selectAll         bool
	limitChunks       int64
	requestTransforms []RequestTransform

	resources []string

	namespace    string
	allNamespace bool
	names        []string

	resourceTuples []resourceTuple

	defaultNamespace bool
	requireNamespace bool

	flatten bool
	latest  bool

	requireObject bool

	singleResourceType bool
	continueOnError    bool

	singleItemImplied bool

	export bool

	schema ContentValidator

	// fakeClientFn is used for testing
	fakeClientFn FakeClientFunc
}

var missingResourceError = fmt.Errorf(`You must provide one or more resources by argument or filename.
Example resource specifications include:
   '-f rsrc.yaml'
   '--filename=rsrc.json'
   '<resource> <name>'
   '<resource>'`)

var LocalResourceError = errors.New(`error: you must specify resources by --filename when --local is set.
Example resource specifications include:
   '-f rsrc.yaml'
   '--filename=rsrc.json'`)

// TODO: expand this to include other errors.
func IsUsageError(err error) bool {
	if err == nil {
		return false
	}
	return err == missingResourceError
}

type FilenameOptions struct {
	Filenames []string
	Kustomize string
	Recursive bool
}

func (o *FilenameOptions) validate() []error {
	var errs []error
	if len(o.Filenames) > 0 && len(o.Kustomize) > 0 {
		errs = append(errs, fmt.Errorf("only one of -f or -k can be specified"))
	}
	if len(o.Kustomize) > 0 && o.Recursive {
		errs = append(errs, fmt.Errorf("the -k flag can't be used with -f or -R"))
	}
	return errs
}

func (o *FilenameOptions) RequireFilenameOrKustomize() error {
	if len(o.Filenames) == 0 && len(o.Kustomize) == 0 {
		return fmt.Errorf("must specify one of -f and -k")
	}
	return nil
}

type resourceTuple struct {
	Resource string
	Name     string
}

type FakeClientFunc func(version schema.GroupVersion) (RESTClient, error)

func NewFakeBuilder(fakeClientFn FakeClientFunc, restMapper RESTMapperFunc, categoryExpander CategoryExpanderFunc) *Builder {
	ret := newBuilder(nil, restMapper, categoryExpander)
	ret.fakeClientFn = fakeClientFn
	return ret
}

// NewBuilder creates a builder that operates on generic objects. At least one of
// internal or unstructured must be specified.
// TODO: Add versioned client (although versioned is still lossy)
// TODO remove internal and unstructured mapper and instead have them set the negotiated serializer for use in the client
func newBuilder(clientConfigFn ClientConfigFunc, restMapper RESTMapperFunc, categoryExpander CategoryExpanderFunc) *Builder {
	return &Builder{
		clientConfigFn:     clientConfigFn,
		restMapperFn:       restMapper,
		categoryExpanderFn: categoryExpander,
		requireObject:      true,
	}
}

func NewBuilder(restClientGetter RESTClientGetter) *Builder {
	categoryExpanderFn := func() (restmapper.CategoryExpander, error) {
		discoveryClient, err := restClientGetter.ToDiscoveryClient()
		if err != nil {
			return nil, err
		}
		return restmapper.NewDiscoveryCategoryExpander(discoveryClient), err
	}

	return newBuilder(
		restClientGetter.ToRESTConfig,
		(&cachingRESTMapperFunc{delegate: restClientGetter.ToRESTMapper}).ToRESTMapper,
		(&cachingCategoryExpanderFunc{delegate: categoryExpanderFn}).ToCategoryExpander,
	)
}

func (b *Builder) Schema(schema ContentValidator) *Builder {
	b.schema = schema
	return b
}

func (b *Builder) AddError(err error) *Builder {
	if err == nil {
		return b
	}
	b.errs = append(b.errs, err)
	return b
}

// FilenameParam groups input in two categories: URLs and files (files, directories, STDIN)
// If enforceNamespace is false, namespaces in the specs will be allowed to
// override the default namespace. If it is true, namespaces that don't match
// will cause an error.
// If ContinueOnError() is set prior to this method, objects on the path that are not
// recognized will be ignored (but logged at V(2)).
func (b *Builder) FilenameParam(enforceNamespace bool, filenameOptions *FilenameOptions) *Builder {
	if errs := filenameOptions.validate(); len(errs) > 0 {
		b.errs = append(b.errs, errs...)
		return b
	}
	recursive := filenameOptions.Recursive
	paths := filenameOptions.Filenames
	for _, s := range paths {
		switch {
		case s == "-":
			b.Stdin()
		case strings.Index(s, "http://") == 0 || strings.Index(s, "https://") == 0:
			url, err := url.Parse(s)
			if err != nil {
				b.errs = append(b.errs, fmt.Errorf("the URL passed to filename %q is not valid: %v", s, err))
				continue
			}
			b.URL(defaultHttpGetAttempts, url)
		default:
			if !recursive {
				b.singleItemImplied = true
			}
			b.Path(recursive, s)
		}
	}
	if filenameOptions.Kustomize != "" {
		b.paths = append(b.paths, &KustomizeVisitor{filenameOptions.Kustomize,
			NewStreamVisitor(nil, b.mapper, filenameOptions.Kustomize, b.schema)})
	}

	if enforceNamespace {
		b.RequireNamespace()
	}

	return b
}

// Unstructured updates the builder so that it will request and send unstructured
// objects. Unstructured objects preserve all fields sent by the server in a map format
// based on the object's JSON structure which means no data is lost when the client
// reads and then writes an object. Use this mode in preference to Internal unless you
// are working with Go types directly.
func (b *Builder) Unstructured() *Builder {
	if b.mapper != nil {
		b.errs = append(b.errs, fmt.Errorf("another mapper was already selected, cannot use unstructured types"))
		return b
	}
	b.objectTyper = unstructuredscheme.NewUnstructuredObjectTyper()
	b.mapper = &mapper{
		localFn:      b.isLocal,
		restMapperFn: b.restMapperFn,
		clientFn:     b.getClient,
		decoder:      unstructured.UnstructuredJSONScheme,
	}

	return b
}

// WithScheme uses the scheme to manage typing, conversion (optional), and decoding.  If decodingVersions
// is empty, then you can end up with internal types.  You have been warned.
func (b *Builder) WithScheme(scheme *runtime.Scheme, decodingVersions ...schema.GroupVersion) *Builder {
	if b.mapper != nil {
		b.errs = append(b.errs, fmt.Errorf("another mapper was already selected, cannot use internal types"))
		return b
	}
	b.objectTyper = scheme
	codecFactory := serializer.NewCodecFactory(scheme)
	negotiatedSerializer := runtime.NegotiatedSerializer(codecFactory)
	// if you specified versions, you're specifying a desire for external types, which you don't want to round-trip through
	// internal types
	if len(decodingVersions) > 0 {
		negotiatedSerializer = codecFactory.WithoutConversion()
	}
	b.negotiatedSerializer = negotiatedSerializer

	b.mapper = &mapper{
		localFn:      b.isLocal,
		restMapperFn: b.restMapperFn,
		clientFn:     b.getClient,
		decoder:      codecFactory.UniversalDecoder(decodingVersions...),
	}

	return b
}

// LocalParam calls Local() if local is true.
func (b *Builder) LocalParam(local bool) *Builder {
	if local {
		b.Local()
	}
	return b
}

// Local will avoid asking the server for results.
func (b *Builder) Local() *Builder {
	b.local = true
	return b
}

func (b *Builder) isLocal() bool {
	return b.local
}

// Mapper returns a copy of the current mapper.
func (b *Builder) Mapper() *mapper {
	mapper := *b.mapper
	return &mapper
}

// URL accepts a number of URLs directly.
func (b *Builder) URL(httpAttemptCount int, urls ...*url.URL) *Builder {
	for _, u := range urls {
		b.paths = append(b.paths, &URLVisitor{
			URL:              u,
			StreamVisitor:    NewStreamVisitor(nil, b.mapper, u.String(), b.schema),
			HttpAttemptCount: httpAttemptCount,
		})
	}
	return b
}

// Stdin will read objects from the standard input. If ContinueOnError() is set
// prior to this method being called, objects in the stream that are unrecognized
// will be ignored (but logged at V(2)).
func (b *Builder) Stdin() *Builder {
	b.stream = true
	b.paths = append(b.paths, FileVisitorForSTDIN(b.mapper, b.schema))
	return b
}

// Stream will read objects from the provided reader, and if an error occurs will
// include the name string in the error message. If ContinueOnError() is set
// prior to this method being called, objects in the stream that are unrecognized
// will be ignored (but logged at V(2)).
func (b *Builder) Stream(r io.Reader, name string) *Builder {
	b.stream = true
	b.paths = append(b.paths, NewStreamVisitor(r, b.mapper, name, b.schema))
	return b
}

// Path accepts a set of paths that may be files, directories (all can containing
// one or more resources). Creates a FileVisitor for each file and then each
// FileVisitor is streaming the content to a StreamVisitor. If ContinueOnError() is set
// prior to this method being called, objects on the path that are unrecognized will be
// ignored (but logged at V(2)).
func (b *Builder) Path(recursive bool, paths ...string) *Builder {
	for _, p := range paths {
		_, err := os.Stat(p)
		if os.IsNotExist(err) {
			b.errs = append(b.errs, fmt.Errorf("the path %q does not exist", p))
			continue
		}
		if err != nil {
			b.errs = append(b.errs, fmt.Errorf("the path %q cannot be accessed: %v", p, err))
			continue
		}

		visitors, err := ExpandPathsToFileVisitors(b.mapper, p, recursive, FileExtensions, b.schema)
		if err != nil {
			b.errs = append(b.errs, fmt.Errorf("error reading %q: %v", p, err))
		}
		if len(visitors) > 1 {
			b.dir = true
		}

		b.paths = append(b.paths, visitors...)
	}
	if len(b.paths) == 0 && len(b.errs) == 0 {
		b.errs = append(b.errs, fmt.Errorf("error reading %v: recognized file extensions are %v", paths, FileExtensions))
	}
	return b
}

// ResourceTypes is a list of types of resources to operate on, when listing objects on
// the server or retrieving objects that match a selector.
func (b *Builder) ResourceTypes(types ...string) *Builder {
	b.resources = append(b.resources, types...)
	return b
}

// ResourceNames accepts a default type and one or more names, and creates tuples of
// resources
func (b *Builder) ResourceNames(resource string, names ...string) *Builder {
	for _, name := range names {
		// See if this input string is of type/name format
		tuple, ok, err := splitResourceTypeName(name)
		if err != nil {
			b.errs = append(b.errs, err)
			return b
		}

		if ok {
			b.resourceTuples = append(b.resourceTuples, tuple)
			continue
		}
		if len(resource) == 0 {
			b.errs = append(b.errs, fmt.Errorf("the argument %q must be RESOURCE/NAME", name))
			continue
		}

		// Use the given default type to create a resource tuple
		b.resourceTuples = append(b.resourceTuples, resourceTuple{Resource: resource, Name: name})
	}
	return b
}

// LabelSelectorParam defines a selector that should be applied to the object types to load.
// This will not affect files loaded from disk or URL. If the parameter is empty it is
// a no-op - to select all resources invoke `b.LabelSelector(labels.Everything.String)`.
func (b *Builder) LabelSelectorParam(s string) *Builder {
	selector := strings.TrimSpace(s)
	if len(selector) == 0 {
		return b
	}
	if b.selectAll {
		b.errs = append(b.errs, fmt.Errorf("found non-empty label selector %q with previously set 'all' parameter. ", s))
		return b
	}
	return b.LabelSelector(selector)
}

// LabelSelector accepts a selector directly and will filter the resulting list by that object.
// Use LabelSelectorParam instead for user input.
func (b *Builder) LabelSelector(selector string) *Builder {
	if len(selector) == 0 {
		return b
	}

	b.labelSelector = &selector
	return b
}

// FieldSelectorParam defines a selector that should be applied to the object types to load.
// This will not affect files loaded from disk or URL. If the parameter is empty it is
// a no-op - to select all resources.
func (b *Builder) FieldSelectorParam(s string) *Builder {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return b
	}
	if b.selectAll {
		b.errs = append(b.errs, fmt.Errorf("found non-empty field selector %q with previously set 'all' parameter. ", s))
		return b
	}
	b.fieldSelector = &s
	return b
}

// ExportParam accepts the export boolean for these resources
func (b *Builder) ExportParam(export bool) *Builder {
	b.export = export
	return b
}

// NamespaceParam accepts the namespace that these resources should be
// considered under from - used by DefaultNamespace() and RequireNamespace()
func (b *Builder) NamespaceParam(namespace string) *Builder {
	b.namespace = namespace
	return b
}

// DefaultNamespace instructs the builder to set the namespace value for any object found
// to NamespaceParam() if empty.
func (b *Builder) DefaultNamespace() *Builder {
	b.defaultNamespace = true
	return b
}

// AllNamespaces instructs the builder to metav1.NamespaceAll as a namespace to request resources
// across all of the namespace. This overrides the namespace set by NamespaceParam().
func (b *Builder) AllNamespaces(allNamespace bool) *Builder {
	if allNamespace {
		b.namespace = metav1.NamespaceAll
	}
	b.allNamespace = allNamespace
	return b
}

// RequireNamespace instructs the builder to set the namespace value for any object found
// to NamespaceParam() if empty, and if the value on the resource does not match
// NamespaceParam() an error will be returned.
func (b *Builder) RequireNamespace() *Builder {
	b.requireNamespace = true
	return b
}

// RequestChunksOf attempts to load responses from the server in batches of size limit
// to avoid long delays loading and transferring very large lists. If unset defaults to
// no chunking.
func (b *Builder) RequestChunksOf(chunkSize int64) *Builder {
	b.limitChunks = chunkSize
	return b
}

// TransformRequests alters API calls made by clients requested from this builder. Pass
// an empty list to clear modifiers.
func (b *Builder) TransformRequests(opts ...RequestTransform) *Builder {
	b.requestTransforms = opts
	return b
}

// SelectEverythingParam
func (b *Builder) SelectAllParam(selectAll bool) *Builder {
	if selectAll && (b.labelSelector != nil || b.fieldSelector != nil) {
		b.errs = append(b.errs, fmt.Errorf("setting 'all' parameter but found a non empty selector. "))
		return b
	}
	b.selectAll = selectAll
	return b
}

// ResourceTypeOrNameArgs indicates that the builder should accept arguments
// of the form `(<type1>[,<type2>,...]|<type> <name1>[,<name2>,...])`. When one argument is
// received, the types provided will be retrieved from the server (and be comma delimited).
// When two or more arguments are received, they must be a single type and resource name(s).
// The allowEmptySelector permits to select all the resources (via Everything func).
func (b *Builder) ResourceTypeOrNameArgs(allowEmptySelector bool, args ...string) *Builder {
	args = normalizeMultipleResourcesArgs(args)
	if ok, err := hasCombinedTypeArgs(args); ok {
		if err != nil {
			b.errs = append(b.errs, err)
			return b
		}
		for _, s := range args {
			tuple, ok, err := splitResourceTypeName(s)
			if err != nil {
				b.errs = append(b.errs, err)
				return b
			}
			if ok {
				b.resourceTuples = append(b.resourceTuples, tuple)
			}
		}
		return b
	}
	if len(args) > 0 {
		// Try replacing aliases only in types
		args[0] = b.ReplaceAliases(args[0])
	}
	switch {
	case len(args) > 2:
		b.names = append(b.names, args[1:]...)
		b.ResourceTypes(SplitResourceArgument(args[0])...)
	case len(args) == 2:
		b.names = append(b.names, args[1])
		b.ResourceTypes(SplitResourceArgument(args[0])...)
	case len(args) == 1:
		b.ResourceTypes(SplitResourceArgument(args[0])...)
		if b.labelSelector == nil && allowEmptySelector {
			selector := labels.Everything().String()
			b.labelSelector = &selector
		}
	case len(args) == 0:
	default:
		b.errs = append(b.errs, fmt.Errorf("arguments must consist of a resource or a resource and name"))
	}
	return b
}

// ReplaceAliases accepts an argument and tries to expand any existing
// aliases found in it
func (b *Builder) ReplaceAliases(input string) string {
	replaced := []string{}
	for _, arg := range strings.Split(input, ",") {
		if b.categoryExpanderFn == nil {
			continue
		}
		categoryExpander, err := b.categoryExpanderFn()
		if err != nil {
			b.AddError(err)
			continue
		}

		if resources, ok := categoryExpander.Expand(arg); ok {
			asStrings := []string{}
			for _, resource := range resources {
				if len(resource.Group) == 0 {
					asStrings = append(asStrings, resource.Resource)
					continue
				}
				asStrings = append(asStrings, resource.Resource+"."+resource.Group)
			}
			arg = strings.Join(asStrings, ",")
		}
		replaced = append(replaced, arg)
	}
	return strings.Join(replaced, ",")
}

func hasCombinedTypeArgs(args []string) (bool, error) {
	hasSlash := 0
	for _, s := range args {
		if strings.Contains(s, "/") {
			hasSlash++
		}
	}
	switch {
	case hasSlash > 0 && hasSlash == len(args):
		return true, nil
	case hasSlash > 0 && hasSlash != len(args):
		baseCmd := "cmd"
		if len(os.Args) > 0 {
			baseCmdSlice := strings.Split(os.Args[0], "/")
			baseCmd = baseCmdSlice[len(baseCmdSlice)-1]
		}
		return true, fmt.Errorf("there is no need to specify a resource type as a separate argument when passing arguments in resource/name form (e.g. '%s get resource/<resource_name>' instead of '%s get resource resource/<resource_name>'", baseCmd, baseCmd)
	default:
		return false, nil
	}
}

// Normalize args convert multiple resources to resource tuples, a,b,c d
// as a transform to a/d b/d c/d
func normalizeMultipleResourcesArgs(args []string) []string {
	if len(args) >= 2 {
		resources := []string{}
		resources = append(resources, SplitResourceArgument(args[0])...)
		if len(resources) > 1 {
			names := []string{}
			names = append(names, args[1:]...)
			newArgs := []string{}
			for _, resource := range resources {
				for _, name := range names {
					newArgs = append(newArgs, strings.Join([]string{resource, name}, "/"))
				}
			}
			return newArgs
		}
	}
	return args
}

// splitResourceTypeName handles type/name resource formats and returns a resource tuple
// (empty or not), whether it successfully found one, and an error
func splitResourceTypeName(s string) (resourceTuple, bool, error) {
	if !strings.Contains(s, "/") {
		return resourceTuple{}, false, nil
	}
	seg := strings.Split(s, "/")
	if len(seg) != 2 {
		return resourceTuple{}, false, fmt.Errorf("arguments in resource/name form may not have more than one slash")
	}
	resource, name := seg[0], seg[1]
	if len(resource) == 0 || len(name) == 0 || len(SplitResourceArgument(resource)) != 1 {
		return resourceTuple{}, false, fmt.Errorf("arguments in resource/name form must have a single resource and name")
	}
	return resourceTuple{Resource: resource, Name: name}, true, nil
}

// Flatten will convert any objects with a field named "Items" that is an array of runtime.Object
// compatible types into individual entries and give them their own items. The original object
// is not passed to any visitors.
func (b *Builder) Flatten() *Builder {
	b.flatten = true
	return b
}

// Latest will fetch the latest copy of any objects loaded from URLs or files from the server.
func (b *Builder) Latest() *Builder {
	b.latest = true
	return b
}

// RequireObject ensures that resulting infos have an object set. If false, resulting info may not have an object set.
func (b *Builder) RequireObject(require bool) *Builder {
	b.requireObject = require
	return b
}

// ContinueOnError will attempt to load and visit as many objects as possible, even if some visits
// return errors or some objects cannot be loaded. The default behavior is to terminate after
// the first error is returned from a VisitorFunc.
func (b *Builder) ContinueOnError() *Builder {
	b.continueOnError = true
	return b
}

// SingleResourceType will cause the builder to error if the user specifies more than a single type
// of resource.
func (b *Builder) SingleResourceType() *Builder {
	b.singleResourceType = true
	return b
}

// mappingFor returns the RESTMapping for the Kind given, or the Kind referenced by the resource.
// Prefers a fully specified GroupVersionResource match. If one is not found, we match on a fully
// specified GroupVersionKind, or fallback to a match on GroupKind.
func (b *Builder) mappingFor(resourceOrKindArg string) (*meta.RESTMapping, error) {
	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(resourceOrKindArg)
	gvk := schema.GroupVersionKind{}
	restMapper, err := b.restMapperFn()
	if err != nil {
		return nil, err
	}

	if fullySpecifiedGVR != nil {
		gvk, _ = restMapper.KindFor(*fullySpecifiedGVR)
	}
	if gvk.Empty() {
		gvk, _ = restMapper.KindFor(groupResource.WithVersion(""))
	}
	if !gvk.Empty() {
		return restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	}

	fullySpecifiedGVK, groupKind := schema.ParseKindArg(resourceOrKindArg)
	if fullySpecifiedGVK == nil {
		gvk := groupKind.WithVersion("")
		fullySpecifiedGVK = &gvk
	}

	if !fullySpecifiedGVK.Empty() {
		if mapping, err := restMapper.RESTMapping(fullySpecifiedGVK.GroupKind(), fullySpecifiedGVK.Version); err == nil {
			return mapping, nil
		}
	}

	mapping, err := restMapper.RESTMapping(groupKind, gvk.Version)
	if err != nil {
		// if we error out here, it is because we could not match a resource or a kind
		// for the given argument. To maintain consistency with previous behavior,
		// announce that a resource type could not be found.
		// if the error is _not_ a *meta.NoKindMatchError, then we had trouble doing discovery,
		// so we should return the original error since it may help a user diagnose what is actually wrong
		if meta.IsNoMatchError(err) {
			return nil, fmt.Errorf("the server doesn't have a resource type %q", groupResource.Resource)
		}
		return nil, err
	}

	return mapping, nil
}

func (b *Builder) resourceMappings() ([]*meta.RESTMapping, error) {
	if len(b.resources) > 1 && b.singleResourceType {
		return nil, fmt.Errorf("you may only specify a single resource type")
	}
	mappings := []*meta.RESTMapping{}
	seen := map[schema.GroupVersionKind]bool{}
	for _, r := range b.resources {
		mapping, err := b.mappingFor(r)
		if err != nil {
			return nil, err
		}
		// This ensures the mappings for resources(shortcuts, plural) unique
		if seen[mapping.GroupVersionKind] {
			continue
		}
		seen[mapping.GroupVersionKind] = true

		mappings = append(mappings, mapping)
	}
	return mappings, nil
}

func (b *Builder) resourceTupleMappings() (map[string]*meta.RESTMapping, error) {
	mappings := make(map[string]*meta.RESTMapping)
	canonical := make(map[schema.GroupVersionResource]struct{})
	for _, r := range b.resourceTuples {
		if _, ok := mappings[r.Resource]; ok {
			continue
		}
		mapping, err := b.mappingFor(r.Resource)
		if err != nil {
			return nil, err
		}

		mappings[r.Resource] = mapping
		canonical[mapping.Resource] = struct{}{}
	}
	if len(canonical) > 1 && b.singleResourceType {
		return nil, fmt.Errorf("you may only specify a single resource type")
	}
	return mappings, nil
}

func (b *Builder) visitorResult() *Result {
	if len(b.errs) > 0 {
		return &Result{err: utilerrors.NewAggregate(b.errs)}
	}

	if b.selectAll {
		selector := labels.Everything().String()
		b.labelSelector = &selector
	}

	// visit items specified by paths
	if len(b.paths) != 0 {
		return b.visitByPaths()
	}

	// visit selectors
	if b.labelSelector != nil || b.fieldSelector != nil {
		return b.visitBySelector()
	}

	// visit items specified by resource and name
	if len(b.resourceTuples) != 0 {
		return b.visitByResource()
	}

	// visit items specified by name
	if len(b.names) != 0 {
		return b.visitByName()
	}

	if len(b.resources) != 0 {
		return &Result{err: fmt.Errorf("resource(s) were provided, but no name, label selector, or --all flag specified")}
	}
	return &Result{err: missingResourceError}
}

func (b *Builder) visitBySelector() *Result {
	result := &Result{
		targetsSingleItems: false,
	}

	if len(b.names) != 0 {
		return result.withError(fmt.Errorf("name cannot be provided when a selector is specified"))
	}
	if len(b.resourceTuples) != 0 {
		return result.withError(fmt.Errorf("selectors and the all flag cannot be used when passing resource/name arguments"))
	}
	if len(b.resources) == 0 {
		return result.withError(fmt.Errorf("at least one resource must be specified to use a selector"))
	}
	mappings, err := b.resourceMappings()
	if err != nil {
		result.err = err
		return result
	}

	var labelSelector, fieldSelector string
	if b.labelSelector != nil {
		labelSelector = *b.labelSelector
	}
	if b.fieldSelector != nil {
		fieldSelector = *b.fieldSelector
	}

	visitors := []Visitor{}
	for _, mapping := range mappings {
		client, err := b.getClient(mapping.GroupVersionKind.GroupVersion())
		if err != nil {
			result.err = err
			return result
		}
		selectorNamespace := b.namespace
		if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
			selectorNamespace = ""
		}
		visitors = append(visitors, NewSelector(client, mapping, selectorNamespace, labelSelector, fieldSelector, b.export, b.limitChunks))
	}
	if b.continueOnError {
		result.visitor = EagerVisitorList(visitors)
	} else {
		result.visitor = VisitorList(visitors)
	}
	result.sources = visitors
	return result
}

func (b *Builder) getClient(gv schema.GroupVersion) (RESTClient, error) {
	var (
		client RESTClient
		err    error
	)

	switch {
	case b.fakeClientFn != nil:
		client, err = b.fakeClientFn(gv)
	case b.negotiatedSerializer != nil:
		client, err = b.clientConfigFn.clientForGroupVersion(gv, b.negotiatedSerializer)
	default:
		client, err = b.clientConfigFn.unstructuredClientForGroupVersion(gv)
	}

	if err != nil {
		return nil, err
	}

	return NewClientWithOptions(client, b.requestTransforms...), nil
}

func (b *Builder) visitByResource() *Result {
	// if b.singleItemImplied is false, this could be by default, so double-check length
	// of resourceTuples to determine if in fact it is singleItemImplied or not
	isSingleItemImplied := b.singleItemImplied
	if !isSingleItemImplied {
		isSingleItemImplied = len(b.resourceTuples) == 1
	}

	result := &Result{
		singleItemImplied:  isSingleItemImplied,
		targetsSingleItems: true,
	}

	if len(b.resources) != 0 {
		return result.withError(fmt.Errorf("you may not specify individual resources and bulk resources in the same call"))
	}

	// retrieve one client for each resource
	mappings, err := b.resourceTupleMappings()
	if err != nil {
		result.err = err
		return result
	}
	clients := make(map[string]RESTClient)
	for _, mapping := range mappings {
		s := fmt.Sprintf("%s/%s", mapping.GroupVersionKind.GroupVersion().String(), mapping.Resource.Resource)
		if _, ok := clients[s]; ok {
			continue
		}
		client, err := b.getClient(mapping.GroupVersionKind.GroupVersion())
		if err != nil {
			result.err = err
			return result
		}
		clients[s] = client
	}

	items := []Visitor{}
	for _, tuple := range b.resourceTuples {
		mapping, ok := mappings[tuple.Resource]
		if !ok {
			return result.withError(fmt.Errorf("resource %q is not recognized: %v", tuple.Resource, mappings))
		}
		s := fmt.Sprintf("%s/%s", mapping.GroupVersionKind.GroupVersion().String(), mapping.Resource.Resource)
		client, ok := clients[s]
		if !ok {
			return result.withError(fmt.Errorf("could not find a client for resource %q", tuple.Resource))
		}

		selectorNamespace := b.namespace
		if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
			selectorNamespace = ""
		} else {
			if len(b.namespace) == 0 {
				errMsg := "namespace may not be empty when retrieving a resource by name"
				if b.allNamespace {
					errMsg = "a resource cannot be retrieved by name across all namespaces"
				}
				return result.withError(fmt.Errorf(errMsg))
			}
		}

		info := &Info{
			Client:    client,
			Mapping:   mapping,
			Namespace: selectorNamespace,
			Name:      tuple.Name,
			Export:    b.export,
		}
		items = append(items, info)
	}

	var visitors Visitor
	if b.continueOnError {
		visitors = EagerVisitorList(items)
	} else {
		visitors = VisitorList(items)
	}
	result.visitor = visitors
	result.sources = items
	return result
}

func (b *Builder) visitByName() *Result {
	result := &Result{
		singleItemImplied:  len(b.names) == 1,
		targetsSingleItems: true,
	}

	if len(b.paths) != 0 {
		return result.withError(fmt.Errorf("when paths, URLs, or stdin is provided as input, you may not specify a resource by arguments as well"))
	}
	if len(b.resources) == 0 {
		return result.withError(fmt.Errorf("you must provide a resource and a resource name together"))
	}
	if len(b.resources) > 1 {
		return result.withError(fmt.Errorf("you must specify only one resource"))
	}

	mappings, err := b.resourceMappings()
	if err != nil {
		result.err = err
		return result
	}
	mapping := mappings[0]

	client, err := b.getClient(mapping.GroupVersionKind.GroupVersion())
	if err != nil {
		result.err = err
		return result
	}

	selectorNamespace := b.namespace
	if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
		selectorNamespace = ""
	} else {
		if len(b.namespace) == 0 {
			errMsg := "namespace may not be empty when retrieving a resource by name"
			if b.allNamespace {
				errMsg = "a resource cannot be retrieved by name across all namespaces"
			}
			return result.withError(fmt.Errorf(errMsg))
		}
	}

	visitors := []Visitor{}
	for _, name := range b.names {
		info := &Info{
			Client:    client,
			Mapping:   mapping,
			Namespace: selectorNamespace,
			Name:      name,
			Export:    b.export,
		}
		visitors = append(visitors, info)
	}
	result.visitor = VisitorList(visitors)
	result.sources = visitors
	return result
}

func (b *Builder) visitByPaths() *Result {
	result := &Result{
		singleItemImplied:  !b.dir && !b.stream && len(b.paths) == 1,
		targetsSingleItems: true,
	}

	if len(b.resources) != 0 {
		return result.withError(fmt.Errorf("when paths, URLs, or stdin is provided as input, you may not specify resource arguments as well"))
	}
	if len(b.names) != 0 {
		return result.withError(fmt.Errorf("name cannot be provided when a path is specified"))
	}
	if len(b.resourceTuples) != 0 {
		return result.withError(fmt.Errorf("resource/name arguments cannot be provided when a path is specified"))
	}

	var visitors Visitor
	if b.continueOnError {
		visitors = EagerVisitorList(b.paths)
	} else {
		visitors = VisitorList(b.paths)
	}

	if b.flatten {
		visitors = NewFlattenListVisitor(visitors, b.objectTyper, b.mapper)
	}

	// only items from disk can be refetched
	if b.latest {
		// must set namespace prior to fetching
		if b.defaultNamespace {
			visitors = NewDecoratedVisitor(visitors, SetNamespace(b.namespace))
		}
		visitors = NewDecoratedVisitor(visitors, RetrieveLatest)
	}
	if b.labelSelector != nil {
		selector, err := labels.Parse(*b.labelSelector)
		if err != nil {
			return result.withError(fmt.Errorf("the provided selector %q is not valid: %v", *b.labelSelector, err))
		}
		visitors = NewFilteredVisitor(visitors, FilterByLabelSelector(selector))
	}
	result.visitor = visitors
	result.sources = b.paths
	return result
}

// Do returns a Result object with a Visitor for the resources identified by the Builder.
// The visitor will respect the error behavior specified by ContinueOnError. Note that stream
// inputs are consumed by the first execution - use Infos() or Object() on the Result to capture a list
// for further iteration.
func (b *Builder) Do() *Result {
	r := b.visitorResult()
	r.mapper = b.Mapper()
	if r.err != nil {
		return r
	}
	if b.flatten {
		r.visitor = NewFlattenListVisitor(r.visitor, b.objectTyper, b.mapper)
	}
	helpers := []VisitorFunc{}
	if b.defaultNamespace {
		helpers = append(helpers, SetNamespace(b.namespace))
	}
	if b.requireNamespace {
		helpers = append(helpers, RequireNamespace(b.namespace))
	}
	helpers = append(helpers, FilterNamespace)
	if b.requireObject {
		helpers = append(helpers, RetrieveLazy)
	}
	if b.continueOnError {
		r.visitor = NewDecoratedVisitor(ContinueOnErrorVisitor{r.visitor}, helpers...)
	} else {
		r.visitor = NewDecoratedVisitor(r.visitor, helpers...)
	}
	return r
}

// SplitResourceArgument splits the argument with commas and returns unique
// strings in the original order.
func SplitResourceArgument(arg string) []string {
	out := []string{}
	set := sets.NewString()
	for _, s := range strings.Split(arg, ",") {
		if set.Has(s) {
			continue
		}
		set.Insert(s)
		out = append(out, s)
	}
	return out
}

// HasNames returns true if the provided args contain resource names
func HasNames(args []string) (bool, error) {
	args = normalizeMultipleResourcesArgs(args)
	hasCombinedTypes, err := hasCombinedTypeArgs(args)
	if err != nil {
		return false, err
	}
	return hasCombinedTypes || len(args) > 1, nil
}

type cachingRESTMapperFunc struct {
	delegate RESTMapperFunc

	lock   sync.Mutex
	cached meta.RESTMapper
}

func (c *cachingRESTMapperFunc) ToRESTMapper() (meta.RESTMapper, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.cached != nil {
		return c.cached, nil
	}

	ret, err := c.delegate()
	if err != nil {
		return nil, err
	}
	c.cached = ret
	return c.cached, nil
}

type cachingCategoryExpanderFunc struct {
	delegate CategoryExpanderFunc

	lock   sync.Mutex
	cached restmapper.CategoryExpander
}

func (c *cachingCategoryExpanderFunc) ToCategoryExpander() (restmapper.CategoryExpander, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.cached != nil {
		return c.cached, nil
	}

	ret, err := c.delegate()
	if err != nil {
		return nil, err
	}
	c.cached = ret
	return c.cached, nil
}
