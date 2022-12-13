/*
Copyright 2019 The Kubernetes Authors.

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

package apply

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/pkg/errors"

	"github.com/jonboulle/clockwork"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/jsonmergepatch"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/kube-openapi/pkg/util/proto"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/openapi"
)

const (
	// maxPatchRetry is the maximum number of conflicts retry for during a patch operation before returning failure
	maxPatchRetry = 5
	// backOffPeriod is the period to back off when apply patch results in error.
	backOffPeriod = 1 * time.Second
	// how many times we can retry before back off
	triesBeforeBackOff = 1
)

var createPatchErrFormat = "creating patch with:\noriginal:\n%s\nmodified:\n%s\ncurrent:\n%s\nfor:"

// Patcher defines options to patch OpenAPI objects.
type Patcher struct {
	Mapping *meta.RESTMapping
	Helper  *resource.Helper

	Overwrite bool
	BackOff   clockwork.Clock

	Force             bool
	CascadingStrategy metav1.DeletionPropagation
	Timeout           time.Duration
	GracePeriod       int

	// If set, forces the patch against a specific resourceVersion
	ResourceVersion *string

	// Number of retries to make if the patch fails with conflict
	Retries int

	OpenapiSchema openapi.Resources
}

func newPatcher(o *ApplyOptions, info *resource.Info, helper *resource.Helper) (*Patcher, error) {
	var openapiSchema openapi.Resources
	if o.OpenAPIPatch {
		openapiSchema = o.OpenAPISchema
	}

	return &Patcher{
		Mapping:           info.Mapping,
		Helper:            helper,
		Overwrite:         o.Overwrite,
		BackOff:           clockwork.NewRealClock(),
		Force:             o.DeleteOptions.ForceDeletion,
		CascadingStrategy: o.DeleteOptions.CascadingStrategy,
		Timeout:           o.DeleteOptions.Timeout,
		GracePeriod:       o.DeleteOptions.GracePeriod,
		OpenapiSchema:     openapiSchema,
		Retries:           maxPatchRetry,
	}, nil
}

func (p *Patcher) delete(namespace, name string) error {
	options := asDeleteOptions(p.CascadingStrategy, p.GracePeriod)
	_, err := p.Helper.DeleteWithOptions(namespace, name, &options)
	return err
}

func (p *Patcher) patchSimple(obj runtime.Object, modified []byte, namespace, name string, errOut io.Writer) ([]byte, runtime.Object, error) {
	// Serialize the current configuration of the object from the server.
	current, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "serializing current configuration from:\n%v\nfor:", obj)
	}

	// Retrieve the original configuration of the object from the annotation.
	original, err := util.GetOriginalConfiguration(obj)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "retrieving original configuration from:\n%v\nfor:", obj)
	}

	var patchType types.PatchType
	var patch []byte

	if p.OpenapiSchema != nil {
		// if openapischema is used, we'll try to get required patch type for this GVK from Open API.
		// if it fails or could not find any patch type, fall back to baked-in patch type determination.
		if patchType, err = p.getPatchTypeFromOpenAPI(p.Mapping.GroupVersionKind); err == nil && patchType == types.StrategicMergePatchType {
			if s := p.findOpenAPIResource(p.Mapping.GroupVersionKind); s != nil {
				patch, err = p.buildStrategicMergeFromOpenAPI(s, original, modified, current)
				if err != nil {
					// Warn user about problem and continue strategic merge patching using builtin types.
					fmt.Fprintf(errOut, "warning: error calculating patch from openapi spec: %v\n", err)
				}
			}
		}
	}

	if patch == nil {
		versionedObj, err := scheme.Scheme.New(p.Mapping.GroupVersionKind)
		if err == nil {
			patchType = types.StrategicMergePatchType
			patch, err = p.buildStrategicMergeFromBuiltins(versionedObj, original, modified, current)
			if err != nil {
				return nil, nil, errors.Wrapf(err, createPatchErrFormat, original, modified, current)
			}
		} else {
			if !runtime.IsNotRegisteredError(err) {
				return nil, nil, errors.Wrapf(err, "getting instance of versioned object for %v:", p.Mapping.GroupVersionKind)
			}

			patchType = types.MergePatchType
			patch, err = p.buildMergePatch(original, modified, current)
			if err != nil {
				return nil, nil, errors.Wrapf(err, createPatchErrFormat, original, modified, current)
			}
		}
	}

	if string(patch) == "{}" {
		return patch, obj, nil
	}

	if p.ResourceVersion != nil {
		patch, err = addResourceVersion(patch, *p.ResourceVersion)
		if err != nil {
			return nil, nil, errors.Wrap(err, "Failed to insert resourceVersion in patch")
		}
	}

	patchedObj, err := p.Helper.Patch(namespace, name, patchType, patch, nil)
	return patch, patchedObj, err
}

// buildMergePatch builds patch according to the JSONMergePatch which is used for
// custom resource definitions.
func (p *Patcher) buildMergePatch(original, modified, current []byte) ([]byte, error) {
	preconditions := []mergepatch.PreconditionFunc{mergepatch.RequireKeyUnchanged("apiVersion"),
		mergepatch.RequireKeyUnchanged("kind"), mergepatch.RequireMetadataKeyUnchanged("name")}
	patch, err := jsonmergepatch.CreateThreeWayJSONMergePatch(original, modified, current, preconditions...)
	if err != nil {
		if mergepatch.IsPreconditionFailed(err) {
			return nil, fmt.Errorf("%s", "At least one of apiVersion, kind and name was changed")
		}
		return nil, err
	}

	return patch, nil
}

// buildStrategicMergeFromOpenAPI builds patch from OpenAPI if it is enabled.
// This is used for core types which is published in openapi.
func (p *Patcher) buildStrategicMergeFromOpenAPI(schema proto.Schema, original, modified, current []byte) ([]byte, error) {
	lookupPatchMeta := strategicpatch.PatchMetaFromOpenAPI{Schema: schema}
	if openapiPatch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.Overwrite); err != nil {
		return nil, err
	} else {
		return openapiPatch, nil
	}
}

// findOpenAPIResource finds schema of GVK in OpenAPI endpoint.
func (p *Patcher) findOpenAPIResource(gvk schema.GroupVersionKind) proto.Schema {
	if p.OpenapiSchema == nil {
		return nil
	}
	return p.OpenapiSchema.LookupResource(gvk)
}

// getPatchTypeFromOpenAPI looks up patch types supported by given GroupVersionKind in Open API.
func (p *Patcher) getPatchTypeFromOpenAPI(gvk schema.GroupVersionKind) (types.PatchType, error) {
	if pc := p.OpenapiSchema.GetConsumes(p.Mapping.GroupVersionKind, "PATCH"); pc != nil {
		for _, c := range pc {
			if c == string(types.StrategicMergePatchType) {
				return types.StrategicMergePatchType, nil
			}
		}

		return types.MergePatchType, nil
	}

	return types.MergePatchType, fmt.Errorf("unable to find any patch type for %s in Open API", gvk)
}

// buildStrategicMergeFromStruct builds patch from struct. This is used when
// openapi endpoint is not working or user disables it by setting openapi-patch flag
// to false.
func (p *Patcher) buildStrategicMergeFromBuiltins(versionedObj runtime.Object, original, modified, current []byte) ([]byte, error) {
	lookupPatchMeta, err := strategicpatch.NewPatchMetaFromStruct(versionedObj)
	if err != nil {
		return nil, err
	}
	patch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.Overwrite)
	if err != nil {
		return nil, err
	}

	return patch, nil
}

// Patch tries to patch an OpenAPI resource. On success, returns the merge patch as well
// the final patched object. On failure, returns an error.
func (p *Patcher) Patch(current runtime.Object, modified []byte, source, namespace, name string, errOut io.Writer) ([]byte, runtime.Object, error) {
	var getErr error
	patchBytes, patchObject, err := p.patchSimple(current, modified, namespace, name, errOut)
	if p.Retries == 0 {
		p.Retries = maxPatchRetry
	}
	for i := 1; i <= p.Retries && apierrors.IsConflict(err); i++ {
		if i > triesBeforeBackOff {
			p.BackOff.Sleep(backOffPeriod)
		}
		current, getErr = p.Helper.Get(namespace, name)
		if getErr != nil {
			return nil, nil, getErr
		}
		patchBytes, patchObject, err = p.patchSimple(current, modified, namespace, name, errOut)
	}
	if err != nil {
		if (apierrors.IsConflict(err) || apierrors.IsInvalid(err)) && p.Force {
			patchBytes, patchObject, err = p.deleteAndCreate(current, modified, namespace, name)
		} else {
			err = cmdutil.AddSourceToErr("patching", source, err)
		}
	}
	return patchBytes, patchObject, err
}

func (p *Patcher) deleteAndCreate(original runtime.Object, modified []byte, namespace, name string) ([]byte, runtime.Object, error) {
	if err := p.delete(namespace, name); err != nil {
		return modified, nil, err
	}
	// TODO: use wait
	if err := wait.PollImmediate(1*time.Second, p.Timeout, func() (bool, error) {
		if _, err := p.Helper.Get(namespace, name); !apierrors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}); err != nil {
		return modified, nil, err
	}
	versionedObject, _, err := unstructured.UnstructuredJSONScheme.Decode(modified, nil, nil)
	if err != nil {
		return modified, nil, err
	}
	createdObject, err := p.Helper.Create(namespace, true, versionedObject)
	if err != nil {
		// restore the original object if we fail to create the new one
		// but still propagate and advertise error to user
		recreated, recreateErr := p.Helper.Create(namespace, true, original)
		if recreateErr != nil {
			err = fmt.Errorf("An error occurred force-replacing the existing object with the newly provided one:\n\n%v.\n\nAdditionally, an error occurred attempting to restore the original object:\n\n%v", err, recreateErr)
		} else {
			createdObject = recreated
		}
	}
	return modified, createdObject, err
}

func addResourceVersion(patch []byte, rv string) ([]byte, error) {
	var patchMap map[string]interface{}
	err := json.Unmarshal(patch, &patchMap)
	if err != nil {
		return nil, err
	}
	u := unstructured.Unstructured{Object: patchMap}
	a, err := meta.Accessor(&u)
	if err != nil {
		return nil, err
	}
	a.SetResourceVersion(rv)

	return json.Marshal(patchMap)
}
