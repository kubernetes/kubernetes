/*
Copyright 2025 The Kubernetes Authors.

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

package audit

import (
	"context"
	"io"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

func TestLogResponseObjectWithPod(t *testing.T) {
	testPod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add core/v1 to scheme: %v", err)
	}
	codecs := serializer.NewCodecFactory(scheme)
	negotiatedSerializer := codecs.WithoutConversion()

	// Create audit context with RequestResponse level
	ctx := WithAuditContext(context.Background())
	ac := AuditContextFrom(ctx)

	captureSink := &capturingAuditSink{}
	if err := ac.Init(RequestAuditConfig{Level: auditinternal.LevelRequestResponse}, captureSink); err != nil {
		t.Fatalf("Failed to initialize audit context: %v", err)
	}

	LogResponseObject(ctx, testPod, schema.GroupVersion{Group: "", Version: "v1"}, negotiatedSerializer)
	ac.ProcessEventStage(ctx, auditinternal.StageResponseComplete)

	if len(captureSink.events) != 1 {
		t.Fatalf("Expected one audit event to be captured, got %d", len(captureSink.events))
	}
	event := captureSink.events[0]
	if event.ResponseObject == nil {
		t.Fatal("Expected ResponseObject to be set, but it was nil")
	}
	if event.ResponseObject.ContentType != runtime.ContentTypeJSON {
		t.Errorf("Expected ContentType to be %q, got %q", runtime.ContentTypeJSON, event.ResponseObject.ContentType)
	}
	if len(event.ResponseObject.Raw) == 0 {
		t.Error("Expected ResponseObject.Raw to contain data, but it was empty")
	}

	responseJSON := string(event.ResponseObject.Raw)
	expectedFields := []string{"test-pod", "test-namespace", "test-container", "test-image"}
	for _, field := range expectedFields {
		if !strings.Contains(responseJSON, field) {
			t.Errorf("Response should contain %q but didn't. Response: %s", field, responseJSON)
		}
	}

	if event.ResponseStatus != nil {
		t.Errorf("Expected ResponseStatus to be nil for regular object, got: %+v", event.ResponseStatus)
	}
}

func TestLogResponseObjectWithStatus(t *testing.T) {
	testCases := []struct {
		name               string
		level              auditinternal.Level
		status             *metav1.Status
		shouldEncode       bool
		expectResponseObj  bool
		expectStatusFields bool
	}{
		{
			name:               "RequestResponse level should encode and log status fields",
			level:              auditinternal.LevelRequestResponse,
			status:             &metav1.Status{Status: "Success", Message: "Test message", Code: 200},
			shouldEncode:       true,
			expectResponseObj:  true,
			expectStatusFields: true,
		},
		{
			name:               "Metadata level should log status fields without encoding",
			level:              auditinternal.LevelMetadata,
			status:             &metav1.Status{Status: "Success", Message: "Test message", Code: 200},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			if err := metav1.AddMetaToScheme(scheme); err != nil {
				t.Fatalf("Failed to add meta to scheme: %v", err)
			}
			scheme.AddKnownTypes(schema.GroupVersion{Version: "v1"}, &metav1.Status{})
			codecs := serializer.NewCodecFactory(scheme)
			negotiatedSerializer := codecs.WithoutConversion()

			ctx := WithAuditContext(context.Background())
			ac := AuditContextFrom(ctx)

			captureSink := &capturingAuditSink{}
			if err := ac.Init(RequestAuditConfig{Level: tc.level}, captureSink); err != nil {
				t.Fatalf("Failed to initialize audit context: %v", err)
			}

			LogResponseObject(ctx, tc.status, schema.GroupVersion{Group: "", Version: "v1"}, negotiatedSerializer)
			ac.ProcessEventStage(ctx, auditinternal.StageResponseComplete)

			if len(captureSink.events) != 1 {
				t.Fatalf("Expected one audit event to be captured, got %d", len(captureSink.events))
			}
			event := captureSink.events[0]

			if tc.expectResponseObj {
				if event.ResponseObject == nil {
					t.Error("Expected ResponseObject to be set, but it was nil")
				}
			} else {
				if event.ResponseObject != nil {
					t.Error("Expected ResponseObject to be nil")
				}
			}

			if tc.expectStatusFields {
				if event.ResponseStatus == nil {
					t.Fatal("Expected ResponseStatus to be set, but it was nil")
				}
				if event.ResponseStatus.Status != tc.status.Status {
					t.Errorf("Expected ResponseStatus.Status to be %q, got %q", tc.status.Status, event.ResponseStatus.Status)
				}
				if event.ResponseStatus.Message != tc.status.Message {
					t.Errorf("Expected ResponseStatus.Message to be %q, got %q", tc.status.Message, event.ResponseStatus.Message)
				}
				if event.ResponseStatus.Code != tc.status.Code {
					t.Errorf("Expected ResponseStatus.Code to be %d, got %d", tc.status.Code, event.ResponseStatus.Code)
				}
			} else if event.ResponseStatus != nil {
				t.Error("Expected ResponseStatus to be nil")
			}
		})
	}
}

func TestLogResponseObjectLevelCheck(t *testing.T) {
	testCases := []struct {
		name               string
		level              auditinternal.Level
		obj                runtime.Object
		shouldEncode       bool
		expectResponseObj  bool
		expectStatusFields bool
	}{
		{
			name:               "None level should not encode or log anything",
			level:              auditinternal.LevelNone,
			obj:                &corev1.Pod{},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: false,
		},
		{
			name:               "Metadata level should not encode or log anything",
			level:              auditinternal.LevelMetadata,
			obj:                &corev1.Pod{},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: false,
		},
		{
			name:  "Metadata level with Status should log status fields without encoding",
			level: auditinternal.LevelMetadata,
			obj: &metav1.Status{
				Status:  "Success",
				Message: "Test message",
				Code:    200,
			},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: true,
		},
		{
			name:               "Request level with Pod should not encode or log",
			level:              auditinternal.LevelRequest,
			obj:                &corev1.Pod{},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: false,
		},
		{
			name:  "Request level with Status should log status fields without encoding",
			level: auditinternal.LevelRequest,
			obj: &metav1.Status{
				Status:  "Success",
				Message: "Test message",
				Code:    200,
			},
			shouldEncode:       false,
			expectResponseObj:  false,
			expectStatusFields: true,
		},
		{
			name:               "RequestResponse level with Pod should encode",
			level:              auditinternal.LevelRequestResponse,
			obj:                &corev1.Pod{},
			shouldEncode:       true,
			expectResponseObj:  true,
			expectStatusFields: false,
		},
		{
			name:  "RequestResponse level with Status should encode and log status fields",
			level: auditinternal.LevelRequestResponse,
			obj: &metav1.Status{
				Status:  "Success",
				Message: "Test message",
				Code:    200,
			},
			shouldEncode:       true,
			expectResponseObj:  true,
			expectStatusFields: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := WithAuditContext(context.Background())
			ac := AuditContextFrom(ctx)

			captureSink := &capturingAuditSink{}
			if err := ac.Init(RequestAuditConfig{Level: tc.level}, captureSink); err != nil {
				t.Fatalf("Failed to initialize audit context: %v", err)
			}

			mockSerializer := &mockNegotiatedSerializer{}
			LogResponseObject(ctx, tc.obj, schema.GroupVersion{Group: "", Version: "v1"}, mockSerializer)
			ac.ProcessEventStage(ctx, auditinternal.StageResponseComplete)

			if mockSerializer.encodeCalled != tc.shouldEncode {
				t.Errorf("Expected encoding to be called: %v, but got: %v", tc.shouldEncode, mockSerializer.encodeCalled)
			}

			if len(captureSink.events) != 1 {
				t.Fatalf("Expected one audit event to be captured, got %d", len(captureSink.events))
			}
			event := captureSink.events[0]

			if tc.expectResponseObj {
				if event.ResponseObject == nil {
					t.Error("Expected ResponseObject to be set, but it was nil")
				}
			} else {
				if event.ResponseObject != nil {
					t.Error("Expected ResponseObject to be nil")
				}
			}

			// Check ResponseStatus for Status objects
			status, isStatus := tc.obj.(*metav1.Status)
			if isStatus && tc.expectStatusFields {
				if event.ResponseStatus == nil {
					t.Error("Expected ResponseStatus to be set for Status object, but it was nil")
				} else {
					if event.ResponseStatus.Status != status.Status {
						t.Errorf("Expected ResponseStatus.Status to be %q, got %q", status.Status, event.ResponseStatus.Status)
					}
					if event.ResponseStatus.Message != status.Message {
						t.Errorf("Expected ResponseStatus.Message to be %q, got %q", status.Message, event.ResponseStatus.Message)
					}
					if event.ResponseStatus.Code != status.Code {
						t.Errorf("Expected ResponseStatus.Code to be %d, got %d", status.Code, event.ResponseStatus.Code)
					}
				}
			} else if event.ResponseStatus != nil {
				t.Error("Expected ResponseStatus to be nil")
			}
		})
	}
}

type mockNegotiatedSerializer struct {
	encodeCalled bool
}

func (m *mockNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{
		{
			MediaType:        runtime.ContentTypeJSON,
			EncodesAsText:    true,
			Serializer:       nil,
			PrettySerializer: nil,
			StreamSerializer: nil,
		},
	}
}

func (m *mockNegotiatedSerializer) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	m.encodeCalled = true
	return &mockEncoder{}
}

func (m *mockNegotiatedSerializer) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return nil
}

type mockEncoder struct{}

func (e *mockEncoder) Encode(obj runtime.Object, w io.Writer) error {
	return nil
}

func (e *mockEncoder) Identifier() runtime.Identifier {
	return runtime.Identifier("mock")
}

type capturingAuditSink struct {
	events []*auditinternal.Event
}

func (s *capturingAuditSink) ProcessEvents(events ...*auditinternal.Event) bool {
	for _, event := range events {
		eventCopy := event.DeepCopy()
		s.events = append(s.events, eventCopy)
	}
	return true
}
