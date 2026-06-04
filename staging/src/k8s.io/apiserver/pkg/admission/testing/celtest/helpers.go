/*
Copyright The Kubernetes Authors.

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

package celtest

import (
	"fmt"
	"reflect"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	admissioncel "k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/authentication/user"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
)

// evaluationInputs holds the converted runtime values needed by ForInput.
type evaluationInputs struct {
	versionedAttr *admission.VersionedAttributes
	request       *admissionv1.AdmissionRequest
	optionalVars  admissioncel.OptionalVariableBindings
	namespace     *corev1.Namespace
}

// buildEvaluationInputs converts an AdmissionInput into the VersionedAttributes,
// AdmissionRequest, and variable bindings required by the CEL evaluator.
// GVK, GVR, operation, and user info are resolved from the request or inferred
// from the object when not explicitly provided.
func buildEvaluationInputs(input *AdmissionInput) (*evaluationInputs, error) {
	if input == nil {
		input = &AdmissionInput{}
	}

	equivalentGVK := resolveEquivalentGVK(input)
	requestGVK := resolveRequestGVK(input.request, equivalentGVK)
	equivalentGVR := resolveEquivalentGVR(input.request, equivalentGVK)
	requestGVR := resolveRequestGVR(input.request, equivalentGVR)

	object, err := convertInputObject(input.object, equivalentGVK)
	if err != nil {
		return nil, fmt.Errorf("converting object: %w", err)
	}
	oldObject, err := convertInputObject(input.oldObject, equivalentGVK)
	if err != nil {
		return nil, fmt.Errorf("converting oldObject: %w", err)
	}
	params, err := convertParamsObject(input.params)
	if err != nil {
		return nil, fmt.Errorf("converting params: %w", err)
	}

	var objectRuntime runtime.Object
	if object != nil {
		objectRuntime = object
	}
	var oldObjectRuntime runtime.Object
	if oldObject != nil {
		oldObjectRuntime = oldObject
	}
	var paramsRuntime runtime.Object
	if params != nil {
		paramsRuntime = params
	}

	name, namespace := resolveNameAndNamespace(input.request, object, oldObject)
	attr := admission.NewAttributesRecord(
		objectRuntime,
		oldObjectRuntime,
		requestGVK,
		namespace,
		name,
		requestGVR,
		resolveSubresource(input.request),
		resolveOperation(input.request, object, oldObject),
		resolveOperationOptions(input.request),
		resolveDryRun(input.request),
		resolveUserInfo(input.request),
	)

	versionedAttr, err := admission.NewVersionedAttributes(attr, equivalentGVK, admission.NewObjectInterfacesFromScheme(runtime.NewScheme()))
	if err != nil {
		return nil, fmt.Errorf("creating versioned admission attributes: %w", err)
	}

	request := admissioncel.CreateAdmissionRequest(attr, metav1GVRFromSchema(equivalentGVR), metav1GVKFromSchema(equivalentGVK))

	return &evaluationInputs{
		versionedAttr: versionedAttr,
		request:       request,
		optionalVars: admissioncel.OptionalVariableBindings{
			VersionedParams: paramsRuntime,
			Authorizer:      input.authorizer,
		},
		namespace: admissioncel.CreateNamespaceObject(input.namespace),
	}, nil
}

func convertInputObject(value interface{}, defaultGVK schema.GroupVersionKind) (*unstructured.Unstructured, error) {
	object, err := convertObjectToUnstructured(value)
	if err != nil {
		return nil, err
	}
	if object == nil {
		return nil, nil
	}
	if gvk := object.GroupVersionKind(); gvk.Empty() {
		object.SetGroupVersionKind(defaultGVK)
	}
	if gvk := object.GroupVersionKind(); gvk.Empty() {
		object.SetGroupVersionKind(defaultObjectGVK())
	}
	return object, nil
}

func convertParamsObject(value interface{}) (*unstructured.Unstructured, error) {
	return convertObjectToUnstructured(value)
}

func convertObjectToUnstructured(value interface{}) (*unstructured.Unstructured, error) {
	if isNilObjectValue(value) {
		return nil, nil
	}
	switch typed := value.(type) {
	case map[string]interface{}:
		return &unstructured.Unstructured{Object: deepCopyMap(typed)}, nil
	case runtime.Object:
		content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typed)
		if err != nil {
			return nil, err
		}
		object := &unstructured.Unstructured{Object: content}
		if gvk := object.GroupVersionKind(); gvk.Empty() {
			if inferred, ok := gvkFromRuntimeObject(typed); ok {
				object.SetGroupVersionKind(inferred)
			}
		}
		return object, nil
	default:
		return nil, fmt.Errorf("unsupported value type %T, must be map[string]interface{} or runtime.Object", value)
	}
}

func resolveEquivalentGVK(input *AdmissionInput) schema.GroupVersionKind {
	if input == nil {
		return defaultObjectGVK()
	}
	if input.request != nil {
		if gvk := schemaGVKFromMeta(input.request.Kind); !gvk.Empty() {
			return gvk
		}
		if input.request.RequestKind != nil {
			if gvk := schemaGVKFromMeta(*input.request.RequestKind); !gvk.Empty() {
				return gvk
			}
		}
	}
	if gvk, ok := gvkFromValue(input.object); ok {
		return gvk
	}
	if gvk, ok := gvkFromValue(input.oldObject); ok {
		return gvk
	}
	return defaultObjectGVK()
}

func resolveRequestGVK(request *admissionv1.AdmissionRequest, equivalentGVK schema.GroupVersionKind) schema.GroupVersionKind {
	if request != nil && request.RequestKind != nil {
		if gvk := schemaGVKFromMeta(*request.RequestKind); !gvk.Empty() {
			return gvk
		}
	}
	return equivalentGVK
}

func resolveEquivalentGVR(request *admissionv1.AdmissionRequest, equivalentGVK schema.GroupVersionKind) schema.GroupVersionResource {
	if request != nil {
		if gvr := schemaGVRFromMeta(request.Resource); !gvr.Empty() {
			return gvr
		}
		if request.RequestResource != nil {
			if gvr := schemaGVRFromMeta(*request.RequestResource); !gvr.Empty() {
				return gvr
			}
		}
	}
	return defaultResourceForGVK(equivalentGVK)
}

func resolveRequestGVR(request *admissionv1.AdmissionRequest, equivalentGVR schema.GroupVersionResource) schema.GroupVersionResource {
	if request != nil && request.RequestResource != nil {
		if gvr := schemaGVRFromMeta(*request.RequestResource); !gvr.Empty() {
			return gvr
		}
	}
	return equivalentGVR
}

func resolveNameAndNamespace(request *admissionv1.AdmissionRequest, object, oldObject *unstructured.Unstructured) (string, string) {
	name := ""
	namespace := ""
	if request != nil {
		name = request.Name
		namespace = request.Namespace
	}
	if name == "" && object != nil {
		name = object.GetName()
	}
	if name == "" && oldObject != nil {
		name = oldObject.GetName()
	}
	if namespace == "" && object != nil {
		namespace = object.GetNamespace()
	}
	if namespace == "" && oldObject != nil {
		namespace = oldObject.GetNamespace()
	}
	return name, namespace
}

func resolveSubresource(request *admissionv1.AdmissionRequest) string {
	if request == nil {
		return ""
	}
	if request.RequestSubResource != "" {
		return request.RequestSubResource
	}
	return request.SubResource
}

// resolveOperation returns the admission operation from the request, or infers
// it from the presence of object and oldObject (Create, Update, or Delete).
func resolveOperation(request *admissionv1.AdmissionRequest, object, oldObject runtime.Object) admission.Operation {
	if request != nil && request.Operation != "" {
		return admission.Operation(request.Operation)
	}
	switch {
	case object == nil && oldObject != nil:
		return admission.Delete
	case object != nil && oldObject != nil:
		return admission.Update
	default:
		return admission.Create
	}
}

func resolveOperationOptions(request *admissionv1.AdmissionRequest) runtime.Object {
	if request == nil {
		return nil
	}
	return request.Options.Object
}

func resolveDryRun(request *admissionv1.AdmissionRequest) bool {
	return request != nil && request.DryRun != nil && *request.DryRun
}

func resolveUserInfo(request *admissionv1.AdmissionRequest) user.Info {
	if request == nil {
		return nil
	}

	if request.UserInfo.Username == "" && request.UserInfo.UID == "" && len(request.UserInfo.Groups) == 0 && len(request.UserInfo.Extra) == 0 {
		return nil
	}
	extra := make(map[string][]string, len(request.UserInfo.Extra))
	for key, value := range request.UserInfo.Extra {
		extra[key] = []string(value)
	}
	return &user.DefaultInfo{
		Name:   request.UserInfo.Username,
		UID:    request.UserInfo.UID,
		Groups: request.UserInfo.Groups,
		Extra:  extra,
	}
}

func gvkFromMap(value map[string]interface{}) (schema.GroupVersionKind, bool) {
	if value == nil {
		return schema.GroupVersionKind{}, false
	}
	gvk := (&unstructured.Unstructured{Object: deepCopyMap(value)}).GroupVersionKind()
	return gvk, !gvk.Empty()
}

func gvkFromValue(value interface{}) (schema.GroupVersionKind, bool) {
	if isNilObjectValue(value) {
		return schema.GroupVersionKind{}, false
	}
	switch typed := value.(type) {
	case map[string]interface{}:
		return gvkFromMap(typed)
	case runtime.Object:
		return gvkFromRuntimeObject(typed)
	default:
		return schema.GroupVersionKind{}, false
	}
}

func gvkFromRuntimeObject(object runtime.Object) (schema.GroupVersionKind, bool) {
	if object == nil {
		return schema.GroupVersionKind{}, false
	}
	if gvk := object.GetObjectKind().GroupVersionKind(); !gvk.Empty() {
		return gvk, true
	}
	// Custom types should set TypeMeta or provide AdmissionInput request kind,
	// since they are not registered in the built-in client-go scheme.
	gvks, _, err := clientgoscheme.Scheme.ObjectKinds(object)
	if err != nil || len(gvks) == 0 || gvks[0].Empty() {
		return schema.GroupVersionKind{}, false
	}
	return gvks[0], true
}

func isNilObjectValue(value interface{}) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

func deepCopyMap(value map[string]interface{}) map[string]interface{} {
	if value == nil {
		return nil
	}
	copyValue, _ := deepCopyValue(value).(map[string]interface{})
	return copyValue
}

// deepCopyValue recursively copies maps and slices. Scalar types (string,
// bool, int64, float64, nil) are returned as-is since they are immutable value
// types in Go. Other reference types (e.g., []byte, structs) are not deep-copied
// and will be shared with the original. This is sufficient for unstructured
// JSON-decoded data which only contains the handled types.
func deepCopyValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		copyValue := make(map[string]interface{}, len(typed))
		for key, item := range typed {
			copyValue[key] = deepCopyValue(item)
		}
		return copyValue
	case []interface{}:
		copyValue := make([]interface{}, len(typed))
		for index, item := range typed {
			copyValue[index] = deepCopyValue(item)
		}
		return copyValue
	default:
		return typed
	}
}

func defaultObjectGVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{Version: "v1", Kind: "Object"}
}

// defaultResourceForGVK derives a resource name from a GVK by delegating to
// meta.UnsafeGuessKindToResource, which is the same heuristic the rest of
// Kubernetes uses (e.g., Endpoints→endpoints, NetworkPolicy→networkpolicies).
// The guess is not exhaustive; supply an explicit Request.Resource for
// non-standard plurals or for plural-equals-singular kinds outside the
// known unpluralized set.
func defaultResourceForGVK(gvk schema.GroupVersionKind) schema.GroupVersionResource {
	if gvk.Kind == "" {
		return schema.GroupVersionResource{Group: gvk.Group, Version: gvk.Version, Resource: "objects"}
	}
	plural, _ := meta.UnsafeGuessKindToResource(gvk)
	return plural
}

func schemaGVKFromMeta(gvk metav1.GroupVersionKind) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind}
}

func schemaGVRFromMeta(gvr metav1.GroupVersionResource) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: gvr.Group, Version: gvr.Version, Resource: gvr.Resource}
}

func metav1GVKFromSchema(gvk schema.GroupVersionKind) metav1.GroupVersionKind {
	return metav1.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind}
}

func metav1GVRFromSchema(gvr schema.GroupVersionResource) metav1.GroupVersionResource {
	return metav1.GroupVersionResource{Group: gvr.Group, Version: gvr.Version, Resource: gvr.Resource}
}
