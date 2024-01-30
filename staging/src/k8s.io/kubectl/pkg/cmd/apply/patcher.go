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
	"k8s.io/client-go/openapi3"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/validation/spec"
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
	// groupVersionKindExtensionKey is the key used to lookup the
	// GroupVersionKind value for an object definition from the
	// definition's "extensions" map.
	groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"
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

	OpenAPIGetter openapi.OpenAPIResourcesGetter
	OpenAPIV3Root openapi3.Root
}

func newPatcher(o *ApplyOptions, info *resource.Info, helper *resource.Helper) (*Patcher, error) {
	var openAPIGetter openapi.OpenAPIResourcesGetter
	var openAPIV3Root openapi3.Root

	if o.OpenAPIPatch {
		openAPIGetter = o.OpenAPIGetter
		openAPIV3Root = o.OpenAPIV3Root
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
		OpenAPIGetter:     openAPIGetter,
		OpenAPIV3Root:     openAPIV3Root,
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

	if p.OpenAPIV3Root != nil {
		gvkSupported, err := p.gvkSupportsPatchOpenAPIV3(p.Mapping.GroupVersionKind)
		if err != nil {
			// Realistically this error logging is not needed (not present in V2),
			// but would help us in debugging if users encounter a problem
			// with OpenAPI V3 not present in V2.
			klog.V(5).Infof("warning: OpenAPI V3 path does not exist - group: %s, version %s, kind %s\n",
				p.Mapping.GroupVersionKind.Group, p.Mapping.GroupVersionKind.Version, p.Mapping.GroupVersionKind.Kind)
		} else if gvkSupported {
			patch, err = p.buildStrategicMergePatchFromOpenAPIV3(original, modified, current)
			if err != nil {
				// Fall back to OpenAPI V2 if there is a problem
				// We should remove the fallback in the future,
				// but for the first release it might be beneficial
				// to fall back to OpenAPI V2 while logging the error
				// and seeing if we get any bug reports.
				fmt.Fprintf(errOut, "warning: error calculating patch from openapi v3 spec: %v\n", err)
			} else {
				patchType = types.StrategicMergePatchType
			}
		} else {
			klog.V(5).Infof("warning: OpenAPI V3 path does not support strategic merge patch - group: %s, version %s, kind %s\n",
				p.Mapping.GroupVersionKind.Group, p.Mapping.GroupVersionKind.Version, p.Mapping.GroupVersionKind.Kind)
		}
	}

	if patch == nil && p.OpenAPIGetter != nil {
		if openAPISchema, err := p.OpenAPIGetter.OpenAPISchema(); err == nil && openAPISchema != nil {
			// if openapischema is used, we'll try to get required patch type for this GVK from Open API.
			// if it fails or could not find any patch type, fall back to baked-in patch type determination.
			if patchType, err = p.getPatchTypeFromOpenAPI(openAPISchema, p.Mapping.GroupVersionKind); err == nil && patchType == types.StrategicMergePatchType {
				patch, err = p.buildStrategicMergeFromOpenAPI(openAPISchema, original, modified, current)
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

// gvkSupportsPatchOpenAPIV3 checks if a particular GVK supports the patch operation.
// It returns an error if the OpenAPI V3 could not be downloaded.
func (p *Patcher) gvkSupportsPatchOpenAPIV3(gvk schema.GroupVersionKind) (bool, error) {
	gvSpec, err := p.OpenAPIV3Root.GVSpec(schema.GroupVersion{
		Group:   p.Mapping.GroupVersionKind.Group,
		Version: p.Mapping.GroupVersionKind.Version,
	})
	if err != nil {
		return false, err
	}
	if gvSpec == nil || gvSpec.Paths == nil || gvSpec.Paths.Paths == nil {
		return false, fmt.Errorf("gvk group: %s, version: %s, kind: %s does not exist for OpenAPI V3", gvk.Group, gvk.Version, gvk.Kind)
	}
	for _, path := range gvSpec.Paths.Paths {
		if path.Patch != nil {
			if gvkMatchesSingle(p.Mapping.GroupVersionKind, path.Patch.Extensions) {
				if path.Patch.RequestBody == nil || path.Patch.RequestBody.Content == nil {
					// GVK exists but does not support requestBody. Indication of malformed OpenAPI.
					return false, nil
				}
				if _, ok := path.Patch.RequestBody.Content["application/strategic-merge-patch+json"]; ok {
					return true, nil
				}
				// GVK exists but strategic-merge-patch is not supported. Likely to be a CRD or aggregated resource.
				return false, nil
			}
		}
	}
	return false, nil
}

func gvkMatchesArray(targetGVK schema.GroupVersionKind, ext spec.Extensions) bool {
	var gvkList []map[string]string
	err := ext.GetObject(groupVersionKindExtensionKey, &gvkList)
	if err != nil {
		return false
	}
	for _, gvkMap := range gvkList {
		if gvkMap["group"] == targetGVK.Group &&
			gvkMap["version"] == targetGVK.Version &&
			gvkMap["kind"] == targetGVK.Kind {
			return true
		}
	}
	return false
}

func gvkMatchesSingle(targetGVK schema.GroupVersionKind, ext spec.Extensions) bool {
	var gvkMap map[string]string
	err := ext.GetObject(groupVersionKindExtensionKey, &gvkMap)
	if err != nil {
		return false
	}
	return gvkMap["group"] == targetGVK.Group &&
		gvkMap["version"] == targetGVK.Version &&
		gvkMap["kind"] == targetGVK.Kind
}

func (p *Patcher) buildStrategicMergePatchFromOpenAPIV3(original, modified, current []byte) ([]byte, error) {
	gvSpec, err := p.OpenAPIV3Root.GVSpec(schema.GroupVersion{
		Group:   p.Mapping.GroupVersionKind.Group,
		Version: p.Mapping.GroupVersionKind.Version,
	})
	if err != nil {
		return nil, err
	}
	if gvSpec == nil || gvSpec.Components == nil {
		return nil, fmt.Errorf("OpenAPI V3 Components is nil")
	}
	for _, c := range gvSpec.Components.Schemas {
		if !gvkMatchesArray(p.Mapping.GroupVersionKind, c.Extensions) {
			continue
		}
		lookupPatchMeta := strategicpatch.PatchMetaFromOpenAPIV3{Schema: c, SchemaList: gvSpec.Components.Schemas}
		if openapiv3Patch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.Overwrite); err != nil {
			return nil, err
		} else {
			return openapiv3Patch, nil
		}

	}
	return nil, nil
}

// buildStrategicMergeFromOpenAPI builds patch from OpenAPI if it is enabled.
// This is used for core types which is published in openapi.
func (p *Patcher) buildStrategicMergeFromOpenAPI(openAPISchema openapi.Resources, original, modified, current []byte) ([]byte, error) {
	schema := openAPISchema.LookupResource(p.Mapping.GroupVersionKind)
	if schema == nil {
		// Missing schema returns nil patch; also no error.
		return nil, nil
	}
	lookupPatchMeta := strategicpatch.PatchMetaFromOpenAPI{Schema: schema}
	if openapiPatch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.Overwrite); err != nil {
		return nil, err
	} else {
		return openapiPatch, nil
	}
}

// getPatchTypeFromOpenAPI looks up patch types supported by given GroupVersionKind in Open API.
func (p *Patcher) getPatchTypeFromOpenAPI(openAPISchema openapi.Resources, gvk schema.GroupVersionKind) (types.PatchType, error) {
	if pc := openAPISchema.GetConsumes(p.Mapping.GroupVersionKind, "PATCH"); pc != nil {
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
