package customresourcevalidation

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"

	configv1 "github.com/openshift/api/config/v1"
)

func TestCustomResourceValidator(t *testing.T) {

	const (
		testGroup    = "config.openshift.io"
		testVersion  = "v1"
		testResource = "images"
		testKind     = "Image"
	)

	var testObjectType *configv1.Image

	testCases := []struct {
		description                  string
		object                       runtime.Object
		objectBytes                  []byte
		oldObject                    runtime.Object
		oldObjectBytes               []byte
		kind                         schema.GroupVersionKind
		namespace                    string
		name                         string
		resource                     schema.GroupVersionResource
		subresource                  string
		operation                    admission.Operation
		userInfo                     user.Info
		expectError                  bool
		expectCreateFuncCalled       bool
		expectUpdateFuncCalled       bool
		expectStatusUpdateFuncCalled bool
		validateFuncErr              bool
		expectedObjectType           interface{}
	}{
		{
			description: "ShouldIgnoreUnknownResource",
			resource: schema.GroupVersionResource{
				Group:    "other_group",
				Version:  "other_version",
				Resource: "other_resource",
			},
		},
		{
			description: "ShouldIgnoreUnknownSubresource",
			subresource: "not_status",
		},
		{
			description: "ShouldIgnoreUnknownSubresource",
			subresource: "not_status",
		},
		{
			description: "UnhandledOperationConnect",
			operation:   admission.Connect,
			expectError: true,
		},
		{
			description: "UnhandledOperationDelete",
			operation:   admission.Delete,
			expectError: true,
		},
		{
			description: "UnhandledKind",
			operation:   admission.Create,
			kind: schema.GroupVersionKind{
				Group:   "other_group",
				Version: "other_version",
				Kind:    "other_resource",
			},
			expectError: true,
		},
		{
			description:            "Create",
			operation:              admission.Create,
			objectBytes:            []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			expectCreateFuncCalled: true,
			expectedObjectType:     testObjectType,
		},
		{
			description: "CreateSubresourceNope",
			operation:   admission.Create,
			subresource: "status",
			objectBytes: []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
		},
		{
			description:            "CreateError",
			operation:              admission.Create,
			objectBytes:            []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			validateFuncErr:        true,
			expectCreateFuncCalled: true,
			expectError:            true,
		},
		{
			description:            "Update",
			operation:              admission.Update,
			objectBytes:            []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			oldObjectBytes:         []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			expectUpdateFuncCalled: true,
			expectedObjectType:     testObjectType,
		},
		{
			description:     "UpdateError",
			operation:       admission.Update,
			objectBytes:     []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			oldObjectBytes:  []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			validateFuncErr: true,
			expectError:     true,
		},
		{
			description:                  "UpdateStatus",
			operation:                    admission.Update,
			subresource:                  "status",
			objectBytes:                  []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			oldObjectBytes:               []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			expectStatusUpdateFuncCalled: true,
			expectedObjectType:           testObjectType,
		},
		{
			description:                  "UpdateStatusError",
			operation:                    admission.Update,
			subresource:                  "status",
			objectBytes:                  []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			oldObjectBytes:               []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v/%v"}`, testKind, testGroup, testVersion)),
			expectStatusUpdateFuncCalled: true,
			validateFuncErr:              true,
			expectError:                  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {

			var createFuncCalled bool
			var updateFuncCalled bool
			var updateStatusFuncCalled bool
			var funcArgObject runtime.Object
			var funcArgOldObject runtime.Object

			handler, err := NewValidator(
				map[schema.GroupResource]bool{
					{Group: testGroup, Resource: testResource}: true,
				},
				map[schema.GroupVersionKind]ObjectValidator{
					{Group: testGroup, Version: testVersion, Kind: testKind}: testValidator{
						validateCreate: func(obj runtime.Object) field.ErrorList {
							createFuncCalled = true
							if tc.validateFuncErr {
								return field.ErrorList{field.InternalError(field.NewPath("test"), errors.New("TEST Error"))}
							}
							funcArgObject = obj
							return nil
						},
						validateUpdate: func(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
							if tc.validateFuncErr {
								return field.ErrorList{field.InternalError(field.NewPath("test"), errors.New("TEST Error"))}
							}
							updateFuncCalled = true
							funcArgObject = obj
							funcArgOldObject = oldObj
							return nil
						},
						validateStatusUpdate: func(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
							updateStatusFuncCalled = true
							if tc.validateFuncErr {
								return field.ErrorList{field.InternalError(field.NewPath("test"), errors.New("TEST Error"))}
							}
							funcArgObject = obj
							funcArgOldObject = oldObj
							return nil
						},
					},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			validator := handler.(admission.ValidationInterface)

			if len(tc.objectBytes) > 0 {
				object, kind, err := unstructured.UnstructuredJSONScheme.Decode(tc.objectBytes, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				tc.object = object.(runtime.Object)
				tc.kind = *kind
			}

			if len(tc.oldObjectBytes) > 0 {
				object, kind, err := unstructured.UnstructuredJSONScheme.Decode(tc.oldObjectBytes, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				tc.oldObject = object.(runtime.Object)
				tc.kind = *kind
			}

			if tc.resource == (schema.GroupVersionResource{}) {
				tc.resource = schema.GroupVersionResource{
					Group:    testGroup,
					Version:  testVersion,
					Resource: testResource,
				}
			}

			attributes := admission.NewAttributesRecord(
				tc.object,
				tc.oldObject,
				tc.kind,
				tc.namespace,
				tc.name,
				tc.resource,
				tc.subresource,
				tc.operation,
				nil,
				false,
				tc.userInfo,
			)

			err = validator.Validate(context.TODO(), attributes, nil)
			switch {
			case tc.expectError && err == nil:
				t.Error("Error expected")
			case !tc.expectError && err != nil:
				t.Errorf("Unexpected error: %v", err)
			}
			if tc.expectCreateFuncCalled != createFuncCalled {
				t.Errorf("ValidateObjCreateFunc called: expected: %v, actual: %v", tc.expectCreateFuncCalled, createFuncCalled)
			}
			if tc.expectUpdateFuncCalled != updateFuncCalled {
				t.Errorf("ValidateObjUpdateFunc called: expected: %v, actual: %v", tc.expectUpdateFuncCalled, updateFuncCalled)
			}
			if tc.expectStatusUpdateFuncCalled != updateStatusFuncCalled {
				t.Errorf("ValidateStatusUpdateFunc called: expected: %v, actual: %v", tc.expectStatusUpdateFuncCalled, updateStatusFuncCalled)
			}
			if reflect.TypeOf(tc.expectedObjectType) != reflect.TypeOf(funcArgObject) {
				t.Errorf("Expected %T, actual %T", tc.expectedObjectType, funcArgObject)
			}
			if (tc.oldObject != nil) && (reflect.TypeOf(tc.expectedObjectType) != reflect.TypeOf(funcArgOldObject)) {
				t.Errorf("Expected %T, actual %T", tc.expectedObjectType, funcArgOldObject)
			}
		})
	}

}

type testValidator struct {
	validateCreate       func(uncastObj runtime.Object) field.ErrorList
	validateUpdate       func(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList
	validateStatusUpdate func(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList
}

func (v testValidator) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	return v.validateCreate(uncastObj)
}

func (v testValidator) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return v.validateUpdate(uncastObj, uncastOldObj)

}

func (v testValidator) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return v.validateStatusUpdate(uncastObj, uncastOldObj)
}
