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

package cmd

import (
	"fmt"
	"log"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// ResourceInfo contains temporary info to execute REST call
type ResourceInfo struct {
	Client    kubectl.RESTClient
	Mapping   *meta.RESTMapping
	Namespace string
	Name      string

	// Optional, this is the most recent value returned by the server if available
	runtime.Object
}

// ResourceVisitor lets clients walk the list of resources
type ResourceVisitor interface {
	Visit(func(*ResourceInfo) error) error
}

type ResourceVisitorList []ResourceVisitor

// Visit implements ResourceVisitor
func (l ResourceVisitorList) Visit(fn func(r *ResourceInfo) error) error {
	for i := range l {
		if err := l[i].Visit(fn); err != nil {
			return err
		}
	}
	return nil
}

func NewResourceInfo(client kubectl.RESTClient, mapping *meta.RESTMapping, namespace, name string) *ResourceInfo {
	return &ResourceInfo{
		Client:    client,
		Mapping:   mapping,
		Namespace: namespace,
		Name:      name,
	}
}

// Visit implements ResourceVisitor
func (r *ResourceInfo) Visit(fn func(r *ResourceInfo) error) error {
	return fn(r)
}

// ResourceSelector is a facade for all the resources fetched via label selector
type ResourceSelector struct {
	Client    kubectl.RESTClient
	Mapping   *meta.RESTMapping
	Namespace string
	Selector  labels.Selector
}

// NewResourceSelector creates a resource selector which hides details of getting items by their label selector.
func NewResourceSelector(client kubectl.RESTClient, mapping *meta.RESTMapping, namespace string, selector labels.Selector) *ResourceSelector {
	return &ResourceSelector{
		Client:    client,
		Mapping:   mapping,
		Namespace: namespace,
		Selector:  selector,
	}
}

// Visit implements ResourceVisitor
func (r *ResourceSelector) Visit(fn func(r *ResourceInfo) error) error {
	list, err := kubectl.NewRESTHelper(r.Client, r.Mapping).List(r.Namespace, r.Selector)
	if err != nil {
		if errors.IsBadRequest(err) || errors.IsNotFound(err) {
			glog.V(2).Infof("Unable to perform a label selector query on %s with labels %s: %v", r.Mapping.Resource, r.Selector, err)
			return nil
		}
		return err
	}
	items, err := runtime.ExtractList(list)
	if err != nil {
		return err
	}
	accessor := meta.NewAccessor()
	for i := range items {
		name, err := accessor.Name(items[i])
		if err != nil {
			// items without names cannot be visited
			glog.V(2).Infof("Found %s with labels %s, but can't access the item by name.", r.Mapping.Resource, r.Selector)
			continue
		}
		item := &ResourceInfo{
			Client:    r.Client,
			Mapping:   r.Mapping,
			Namespace: r.Namespace,
			Name:      name,
			Object:    items[i],
		}
		if err := fn(item); err != nil {
			if errors.IsNotFound(err) {
				glog.V(2).Infof("Found %s named %q, but can't be accessed now: %v", r.Mapping.Resource, name, err)
				return nil
			}
			log.Printf("got error for resource %s: %v", r.Mapping.Resource, err)
			return err
		}
	}
	return nil
}

// ResourcesFromArgsOrFile computes a list of Resources by extracting info from filename or args. It will
// handle label selectors provided.
func ResourcesFromArgsOrFile(
	cmd *cobra.Command,
	args []string,
	filename, selector string,
	typer runtime.ObjectTyper,
	mapper meta.RESTMapper,
	clientBuilder func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.RESTClient, error),
	schema validation.Schema,
) ResourceVisitor {

	// handling filename & resource id
	if len(selector) == 0 {
		mapping, namespace, name := ResourceFromArgsOrFile(cmd, args, filename, typer, mapper, schema)
		client, err := clientBuilder(cmd, mapping)
		checkErr(err)

		return NewResourceInfo(client, mapping, namespace, name)
	}

	labelSelector, err := labels.ParseSelector(selector)
	checkErr(err)

	namespace := GetKubeNamespace(cmd)
	visitors := ResourceVisitorList{}

	if len(args) != 1 {
		usageError(cmd, "Must specify the type of resource")
	}
	types := SplitResourceArgument(args[0])
	for _, arg := range types {
		resource := kubectl.ExpandResourceShortcut(arg)
		if len(resource) == 0 {
			usageError(cmd, "Unknown resource %s", resource)
		}
		version, kind, err := mapper.VersionAndKindForResource(resource)
		checkErr(err)

		mapping, err := mapper.RESTMapping(version, kind)
		checkErr(err)

		client, err := clientBuilder(cmd, mapping)
		checkErr(err)

		visitors = append(visitors, NewResourceSelector(client, mapping, namespace, labelSelector))
	}
	return visitors
}

// ResourceFromArgsOrFile expects two arguments or a valid file with a given type, and extracts
// the fields necessary to uniquely locate a resource. Displays a usageError if that contract is
// not satisfied, or a generic error if any other problems occur.
func ResourceFromArgsOrFile(cmd *cobra.Command, args []string, filename string, typer runtime.ObjectTyper, mapper meta.RESTMapper, schema validation.Schema) (mapping *meta.RESTMapping, namespace, name string) {
	// If command line args are passed in, use those preferentially.
	if len(args) > 0 && len(args) != 2 {
		usageError(cmd, "If passing in command line parameters, must be resource and name")
	}

	if len(args) == 2 {
		resource := kubectl.ExpandResourceShortcut(args[0])
		namespace = GetKubeNamespace(cmd)
		name = args[1]
		if len(name) == 0 || len(resource) == 0 {
			usageError(cmd, "Must specify filename or command line params")
		}

		defaultVersion, kind, err := mapper.VersionAndKindForResource(resource)
		if err != nil {
			// The error returned by mapper is "no resource defined", which is a usage error
			usageError(cmd, err.Error())
		}
		version := GetFlagString(cmd, "api-version")
		mapping, err = mapper.RESTMapping(kind, version, defaultVersion)
		checkErr(err)
		return
	}

	if len(filename) == 0 {
		usageError(cmd, "Must specify filename or command line params")
	}

	mapping, namespace, name, _ = ResourceFromFile(filename, typer, mapper, schema)
	if len(name) == 0 {
		checkErr(fmt.Errorf("the resource in the provided file has no name (or ID) defined"))
	}

	return
}

// ResourceFromArgs expects two arguments with a given type, and extracts the fields necessary
// to uniquely locate a resource. Displays a usageError if that contract is not satisfied, or
// a generic error if any other problems occur.
func ResourceFromArgs(cmd *cobra.Command, args []string, mapper meta.RESTMapper) (mapping *meta.RESTMapping, namespace, name string) {
	if len(args) != 2 {
		usageError(cmd, "Must provide resource and name command line params")
	}

	resource := kubectl.ExpandResourceShortcut(args[0])
	namespace = GetKubeNamespace(cmd)
	name = args[1]
	if len(name) == 0 || len(resource) == 0 {
		usageError(cmd, "Must provide resource and name command line params")
	}

	version, kind, err := mapper.VersionAndKindForResource(resource)
	checkErr(err)

	mapping, err = mapper.RESTMapping(kind, version)
	checkErr(err)
	return
}

// ResourceFromArgs expects two arguments with a given type, and extracts the fields necessary
// to uniquely locate a resource. Displays a usageError if that contract is not satisfied, or
// a generic error if any other problems occur.
func ResourceOrTypeFromArgs(cmd *cobra.Command, args []string, mapper meta.RESTMapper) (mapping *meta.RESTMapping, namespace, name string) {
	if len(args) == 0 || len(args) > 2 {
		usageError(cmd, "Must provide resource or a resource and name as command line params")
	}

	resource := kubectl.ExpandResourceShortcut(args[0])
	if len(resource) == 0 {
		usageError(cmd, "Must provide resource or a resource and name as command line params")
	}

	namespace = GetKubeNamespace(cmd)
	if len(args) == 2 {
		name = args[1]
		if len(name) == 0 {
			usageError(cmd, "Must provide resource or a resource and name as command line params")
		}
	}

	defaultVersion, kind, err := mapper.VersionAndKindForResource(resource)
	checkErr(err)

	version := GetFlagString(cmd, "api-version")
	mapping, err = mapper.RESTMapping(kind, version, defaultVersion)
	checkErr(err)

	return
}

// ResourceFromFile retrieves the name and namespace from a valid file. If the file does not
// resolve to a known type an error is returned. The returned mapping can be used to determine
// the correct REST endpoint to modify this resource with.
func ResourceFromFile(filename string, typer runtime.ObjectTyper, mapper meta.RESTMapper, schema validation.Schema) (mapping *meta.RESTMapping, namespace, name string, data []byte) {
	configData, err := ReadConfigData(filename)
	checkErr(err)
	data = configData

	version, kind, err := typer.DataVersionAndKind(data)
	checkErr(err)

	// TODO: allow unversioned objects?
	if len(version) == 0 {
		checkErr(fmt.Errorf("the resource in the provided file has no apiVersion defined"))
	}

	err = schema.ValidateBytes(data)
	checkErr(err)

	mapping, err = mapper.RESTMapping(kind, version)
	checkErr(err)

	obj, err := mapping.Codec.Decode(data)
	checkErr(err)

	meta := mapping.MetadataAccessor
	namespace, err = meta.Namespace(obj)
	checkErr(err)
	name, err = meta.Name(obj)
	checkErr(err)

	return
}

// CompareNamespaceFromFile returns an error if the namespace the user has provided on the CLI
// or via the default namespace file does not match the namespace of an input file. This
// prevents a user from unintentionally updating the wrong namespace.
func CompareNamespaceFromFile(cmd *cobra.Command, namespace string) error {
	defaultNamespace := GetKubeNamespace(cmd)
	if len(namespace) > 0 {
		if defaultNamespace != namespace {
			return fmt.Errorf("the namespace from the provided file %q does not match the namespace %q. You must pass '--namespace=%s' to perform this operation.", namespace, defaultNamespace, namespace)
		}
	}
	return nil
}

func SplitResourceArgument(arg string) []string {
	set := util.NewStringSet()
	set.Insert(strings.Split(arg, ",")...)
	return set.List()
}
