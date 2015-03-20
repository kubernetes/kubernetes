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

package util

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// ResourceFromArgs expects two arguments with a given type, and extracts the fields necessary
// to uniquely locate a resource. Displays a UsageError if that contract is not satisfied, or
// a generic error if any other problems occur.
// DEPRECATED: Use resource.Builder
func ResourceFromArgs(cmd *cobra.Command, args []string, mapper meta.RESTMapper, cmdNamespace string) (mapping *meta.RESTMapping, namespace, name string, err error) {
	if len(args) != 2 {
		err = UsageError(cmd, "Must provide resource and name command line params")
		return
	}

	resource := args[0]
	namespace = cmdNamespace
	name = args[1]
	if len(name) == 0 || len(resource) == 0 {
		err = UsageError(cmd, "Must provide resource and name command line params")
		return
	}

	version, kind, err := mapper.VersionAndKindForResource(resource)
	if err != nil {
		return
	}

	mapping, err = mapper.RESTMapping(kind, version)
	return
}

// ResourceFromFile retrieves the name and namespace from a valid file. If the file does not
// resolve to a known type an error is returned. The returned mapping can be used to determine
// the correct REST endpoint to modify this resource with.
// DEPRECATED: Use resource.Builder
func ResourceFromFile(filename string, typer runtime.ObjectTyper, mapper meta.RESTMapper, schema validation.Schema, cmdVersion string) (mapping *meta.RESTMapping, namespace, name string, data []byte, err error) {
	data, err = ReadConfigData(filename)
	if err != nil {
		return
	}

	objVersion, kind, err := typer.DataVersionAndKind(data)
	if err != nil {
		return
	}

	// TODO: allow unversioned objects?
	if len(objVersion) == 0 {
		err = fmt.Errorf("the resource in the provided file has no apiVersion defined")
	}

	err = schema.ValidateBytes(data)
	if err != nil {
		return
	}

	// decode using the version stored with the object (allows codec to vary across versions)
	mapping, err = mapper.RESTMapping(kind, objVersion)
	if err != nil {
		return
	}

	obj, err := mapping.Codec.Decode(data)
	if err != nil {
		return
	}

	meta := mapping.MetadataAccessor
	namespace, err = meta.Namespace(obj)
	if err != nil {
		return
	}
	name, err = meta.Name(obj)
	if err != nil {
		return
	}

	// if the preferred API version differs, get a different mapper
	if cmdVersion != objVersion {
		mapping, err = mapper.RESTMapping(kind, cmdVersion)
	}
	return
}

// CompareNamespace returns an error if the namespace the user has provided on the CLI
// or via the default namespace file does not match the namespace of an input file. This
// prevents a user from unintentionally updating the wrong namespace.
// DEPRECATED: Use resource.Builder
func CompareNamespace(defaultNamespace, namespace string) error {
	if len(namespace) > 0 {
		if defaultNamespace != namespace {
			return fmt.Errorf("the namespace from the provided file %q does not match the namespace %q. You must pass '--namespace=%s' to perform this operation.", namespace, defaultNamespace, namespace)
		}
	}
	return nil
}
