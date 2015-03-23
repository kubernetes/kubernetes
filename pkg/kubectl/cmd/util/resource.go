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
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
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
