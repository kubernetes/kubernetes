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
	"io"
	"net/url"
	"os"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"
)

var FileExtensions = []string{".json", ".yaml", ".yml"}
var InputExtensions = append(FileExtensions, "stdin")

const defaultHttpGetAttempts int = 3

// Builder provides convenience functions for taking arguments and parameters
// from the command line and converting them to a list of resources to iterate
// over using the Visitor interface.
type Builder struct {
	mapper *Mapper

	errs []error

	paths  []Visitor
	stream bool
	dir    bool

	selector  labels.Selector
	selectAll bool

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

	singular bool

	export bool

	schema validation.Schema
}

var missingResourceError = fmt.Errorf(`You must provide one or more resources by argument or filename.
Example resource specifications include:
   '-f rsrc.yaml'
   '--filename=rsrc.json'
   'pods my-pod'
   'services'`)

// TODO: expand this to include other errors.
func IsUsageError(err error) bool {
	if err == nil {
		return false
	}
	return err == missingResourceError
}

type FilenameOptions struct {
	Filenames []string
	Recursive bool
}

type resourceTuple struct {
	Resource string
	Name     string
}

// NewBuilder creates a builder that operates on generic objects.
func NewBuilder(mapper meta.RESTMapper, typer runtime.ObjectTyper, clientMapper ClientMapper, decoder runtime.Decoder) *Builder {
	return &Builder{
		mapper:        &Mapper{typer, mapper, clientMapper, decoder},
		requireObject: true,
	}
}

func (b *Builder) Schema(schema validation.Schema) *Builder {
	b.schema = schema
	return b
}

// FilenameParam groups input in two categories: URLs and files (files, directories, STDIN)
// If enforceNamespace is false, namespaces in the specs will be allowed to
// override the default namespace. If it is true, namespaces that don't match
// will cause an error.
// If ContinueOnError() is set prior to this method, objects on the path that are not
// recognized will be ignored (but logged at V(2)).
func (b *Builder) FilenameParam(enforceNamespace bool, filenameOptions *FilenameOptions) *Builder {
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
				b.singular = true
			}
			b.Path(recursive, s)
		}
	}

	if enforceNamespace {
		b.RequireNamespace()
	}

	return b
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

// SelectorParam defines a selector that should be applied to the object types to load.
// This will not affect files loaded from disk or URL. If the parameter is empty it is
// a no-op - to select all resources invoke `b.Selector(labels.Everything)`.
func (b *Builder) SelectorParam(s string) *Builder {
	selector, err := labels.Parse(s)
	if err != nil {
		b.errs = append(b.errs, fmt.Errorf("the provided selector %q is not valid: %v", s, err))
		return b
	}
	if selector.Empty() {
		return b
	}
	if b.selectAll {
		b.errs = append(b.errs, fmt.Errorf("found non empty selector %q with previously set 'all' parameter. ", s))
		return b
	}
	return b.Selector(selector)
}

// Selector accepts a selector directly, and if non nil will trigger a list action.
func (b *Builder) Selector(selector labels.Selector) *Builder {
	b.selector = selector
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

// AllNamespaces instructs the builder to use NamespaceAll as a namespace to request resources
// across all of the namespace. This overrides the namespace set by NamespaceParam().
func (b *Builder) AllNamespaces(allNamespace bool) *Builder {
	if allNamespace {
		b.namespace = api.NamespaceAll
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

// SelectEverythingParam
func (b *Builder) SelectAllParam(selectAll bool) *Builder {
	if selectAll && b.selector != nil {
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
		if b.selector == nil && allowEmptySelector {
			b.selector = labels.Everything()
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
		if aliases, ok := b.mapper.AliasesForResource(arg); ok {
			arg = strings.Join(aliases, ",")
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

// mappingFor returns the RESTMapping for the Kind referenced by the resource.
// prefers a fully specified GroupVersionResource match.  If we don't have one match on GroupResource
func (b *Builder) mappingFor(resourceArg string) (*meta.RESTMapping, error) {
	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(resourceArg)
	gvk := schema.GroupVersionKind{}
	if fullySpecifiedGVR != nil {
		gvk, _ = b.mapper.KindFor(*fullySpecifiedGVR)
	}
	if gvk.Empty() {
		var err error
		gvk, err = b.mapper.KindFor(groupResource.WithVersion(""))
		if err != nil {
			return nil, err
		}
	}

	return b.mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
}

func (b *Builder) resourceMappings() ([]*meta.RESTMapping, error) {
	if len(b.resources) > 1 && b.singleResourceType {
		return nil, fmt.Errorf("you may only specify a single resource type")
	}
	mappings := []*meta.RESTMapping{}
	for _, r := range b.resources {
		mapping, err := b.mappingFor(r)
		if err != nil {
			return nil, err
		}

		mappings = append(mappings, mapping)
	}
	return mappings, nil
}

func (b *Builder) resourceTupleMappings() (map[string]*meta.RESTMapping, error) {
	mappings := make(map[string]*meta.RESTMapping)
	canonical := make(map[string]struct{})
	for _, r := range b.resourceTuples {
		if _, ok := mappings[r.Resource]; ok {
			continue
		}
		mapping, err := b.mappingFor(r.Resource)
		if err != nil {
			return nil, err
		}

		mappings[mapping.Resource] = mapping
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
		b.selector = labels.Everything()
	}

	// visit items specified by paths
	if len(b.paths) != 0 {
		return b.visitByPaths()
	}

	// visit selectors
	if b.selector != nil {
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
	if len(b.names) != 0 {
		return &Result{err: fmt.Errorf("name cannot be provided when a selector is specified")}
	}
	if len(b.resourceTuples) != 0 {
		return &Result{err: fmt.Errorf("selectors and the all flag cannot be used when passing resource/name arguments")}
	}
	if len(b.resources) == 0 {
		return &Result{err: fmt.Errorf("at least one resource must be specified to use a selector")}
	}
	mappings, err := b.resourceMappings()
	if err != nil {
		return &Result{err: err}
	}

	visitors := []Visitor{}
	for _, mapping := range mappings {
		client, err := b.mapper.ClientForMapping(mapping)
		if err != nil {
			return &Result{err: err}
		}
		selectorNamespace := b.namespace
		if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
			selectorNamespace = ""
		}
		visitors = append(visitors, NewSelector(client, mapping, selectorNamespace, b.selector, b.export))
	}
	if b.continueOnError {
		return &Result{visitor: EagerVisitorList(visitors), sources: visitors}
	}
	return &Result{visitor: VisitorList(visitors), sources: visitors}
}

func (b *Builder) visitByResource() *Result {
	// if b.singular is false, this could be by default, so double-check length
	// of resourceTuples to determine if in fact it is singular or not
	isSingular := b.singular
	if !isSingular {
		isSingular = len(b.resourceTuples) == 1
	}

	if len(b.resources) != 0 {
		return &Result{singular: isSingular, err: fmt.Errorf("you may not specify individual resources and bulk resources in the same call")}
	}

	// retrieve one client for each resource
	mappings, err := b.resourceTupleMappings()
	if err != nil {
		return &Result{singular: isSingular, err: err}
	}
	clients := make(map[string]RESTClient)
	for _, mapping := range mappings {
		s := fmt.Sprintf("%s/%s", mapping.GroupVersionKind.GroupVersion().String(), mapping.Resource)
		if _, ok := clients[s]; ok {
			continue
		}
		client, err := b.mapper.ClientForMapping(mapping)
		if err != nil {
			return &Result{err: err}
		}
		clients[s] = client
	}

	items := []Visitor{}
	for _, tuple := range b.resourceTuples {
		mapping, ok := mappings[tuple.Resource]
		if !ok {
			return &Result{singular: isSingular, err: fmt.Errorf("resource %q is not recognized: %v", tuple.Resource, mappings)}
		}
		s := fmt.Sprintf("%s/%s", mapping.GroupVersionKind.GroupVersion().String(), mapping.Resource)
		client, ok := clients[s]
		if !ok {
			return &Result{singular: isSingular, err: fmt.Errorf("could not find a client for resource %q", tuple.Resource)}
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
				return &Result{singular: isSingular, err: fmt.Errorf(errMsg)}
			}
		}

		info := NewInfo(client, mapping, selectorNamespace, tuple.Name, b.export)
		items = append(items, info)
	}

	var visitors Visitor
	if b.continueOnError {
		visitors = EagerVisitorList(items)
	} else {
		visitors = VisitorList(items)
	}
	return &Result{singular: isSingular, visitor: visitors, sources: items}
}

func (b *Builder) visitByName() *Result {
	isSingular := len(b.names) == 1

	if len(b.paths) != 0 {
		return &Result{singular: isSingular, err: fmt.Errorf("when paths, URLs, or stdin is provided as input, you may not specify a resource by arguments as well")}
	}
	if len(b.resources) == 0 {
		return &Result{singular: isSingular, err: fmt.Errorf("you must provide a resource and a resource name together")}
	}
	if len(b.resources) > 1 {
		return &Result{singular: isSingular, err: fmt.Errorf("you must specify only one resource")}
	}

	mappings, err := b.resourceMappings()
	if err != nil {
		return &Result{singular: isSingular, err: err}
	}
	mapping := mappings[0]

	client, err := b.mapper.ClientForMapping(mapping)
	if err != nil {
		return &Result{err: err}
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
			return &Result{singular: isSingular, err: fmt.Errorf(errMsg)}
		}
	}

	visitors := []Visitor{}
	for _, name := range b.names {
		info := NewInfo(client, mapping, selectorNamespace, name, b.export)
		visitors = append(visitors, info)
	}
	return &Result{singular: isSingular, visitor: VisitorList(visitors), sources: visitors}
}

func (b *Builder) visitByPaths() *Result {
	singular := !b.dir && !b.stream && len(b.paths) == 1
	if len(b.resources) != 0 {
		return &Result{singular: singular, err: fmt.Errorf("when paths, URLs, or stdin is provided as input, you may not specify resource arguments as well")}
	}
	if len(b.names) != 0 {
		return &Result{err: fmt.Errorf("name cannot be provided when a path is specified")}
	}
	if len(b.resourceTuples) != 0 {
		return &Result{err: fmt.Errorf("resource/name arguments cannot be provided when a path is specified")}
	}

	var visitors Visitor
	if b.continueOnError {
		visitors = EagerVisitorList(b.paths)
	} else {
		visitors = VisitorList(b.paths)
	}

	// only items from disk can be refetched
	if b.latest {
		// must flatten lists prior to fetching
		if b.flatten {
			visitors = NewFlattenListVisitor(visitors, b.mapper)
		}
		// must set namespace prior to fetching
		if b.defaultNamespace {
			visitors = NewDecoratedVisitor(visitors, SetNamespace(b.namespace))
		}
		visitors = NewDecoratedVisitor(visitors, RetrieveLatest)
	}
	if b.selector != nil {
		visitors = NewFilteredVisitor(visitors, FilterBySelector(b.selector))
	}
	return &Result{singular: singular, visitor: visitors, sources: b.paths}
}

// Do returns a Result object with a Visitor for the resources identified by the Builder.
// The visitor will respect the error behavior specified by ContinueOnError. Note that stream
// inputs are consumed by the first execution - use Infos() or Object() on the Result to capture a list
// for further iteration.
func (b *Builder) Do() *Result {
	r := b.visitorResult()
	if r.err != nil {
		return r
	}
	if b.flatten {
		r.visitor = NewFlattenListVisitor(r.visitor, b.mapper)
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
	r.visitor = NewDecoratedVisitor(r.visitor, helpers...)
	if b.continueOnError {
		r.visitor = ContinueOnErrorVisitor{r.visitor}
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

// MultipleTypesRequested returns true if the provided args contain multiple resource kinds
func MultipleTypesRequested(args []string) bool {
	args = normalizeMultipleResourcesArgs(args)
	rKinds := sets.NewString()
	for _, arg := range args {
		if arg == "all" {
			return true
		}
		rTuple, found, err := splitResourceTypeName(arg)
		if err != nil {
			continue
		}

		// if tuple not found, assume arg is of the form "type1,type2,...".
		// Since SplitResourceArgument returns a unique list of kinds,
		// return true here if len(uniqueList) > 1
		if !found {
			if strings.Contains(arg, ",") {
				splitArgs := SplitResourceArgument(arg)
				if len(splitArgs) > 1 {
					return true
				}
			}
			continue
		}
		if rKinds.Has(rTuple.Resource) {
			continue
		}
		rKinds.Insert(rTuple.Resource)
	}
	return (rKinds.Len() > 1)
}
