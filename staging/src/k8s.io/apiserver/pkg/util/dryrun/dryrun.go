/*
Copyright 2018 The Kubernetes Authors.

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

package dryrun

import (
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// IsDryRun returns true if the DryRun flag is an actual dry-run.
func IsDryRun(flag []string) bool {
	return len(flag) > 0
}

// ResetMetadata resets metadata fields that are not allowed to be set by dry-run.
func ResetMetadata(originalObj, newObj runtime.Object) error {
	originalObjMeta, err := meta.Accessor(originalObj)
	if err != nil {
		return errors.NewInternalError(err)
	}
	newObjMeta, err := meta.Accessor(newObj)
	if err != nil {
		return errors.NewInternalError(err)
	}
	// If a resource is created with dry-run enabled where generateName is set, the
	// store will set the name to the generated name. We need to reset the name and restore
	// the generateName metadata fields in order for the returned object to match the intent
	// of the original template.
	if originalObjMeta.GetGenerateName() != "" {
		newObjMeta.SetName("")
	}
	newObjMeta.SetGenerateName(originalObjMeta.GetGenerateName())
	// If UID is set in the dry-run output then that output cannot be used to create a resource. Reset
	// the UID to allow the output to be used to create resources.
	newObjMeta.SetUID("")
	// If the resourceVersion is set in the dry-run output then that output cannot be used to create
	// a resource. Reset the resourceVersion to allow the output to be used to create resources.
	newObjMeta.SetResourceVersion("")

	return nil
}
