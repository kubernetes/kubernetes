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

package handlers

import (
	"context"
	"io"
	"testing"

	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditapis "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/utils/pointer"
)

type mockCodecs struct {
	serializer.CodecFactory
	err error
}

type mockCodec struct {
	runtime.Codec
	codecs *mockCodecs
}

func (p mockCodec) Encode(obj runtime.Object, w io.Writer) error {
	err := p.Codec.Encode(obj, w)
	p.codecs.err = err
	return err
}

func (s *mockCodecs) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	out := s.CodecFactory.CodecForVersions(encoder, nil, gv, nil)
	return &mockCodec{
		Codec:  out,
		codecs: s,
	}
}

func TestDeleteResourceAuditLogRequestObject(t *testing.T) {

	ctx := audit.WithAuditContext(context.TODO(), &audit.AuditContext{
		Event: &auditapis.Event{
			Level: auditapis.LevelRequestResponse,
		},
	})

	policy := metav1.DeletePropagationBackground
	deleteOption := &metav1.DeleteOptions{
		GracePeriodSeconds: pointer.Int64Ptr(30),
		PropagationPolicy:  &policy,
	}

	fakeCorev1GroupVersion := schema.GroupVersion{
		Group:   "",
		Version: "v1",
	}
	testScheme := runtime.NewScheme()
	metav1.AddToGroupVersion(testScheme, fakeCorev1GroupVersion)
	testCodec := serializer.NewCodecFactory(testScheme)

	tests := []struct {
		name       string
		object     runtime.Object
		gv         schema.GroupVersion
		serializer serializer.CodecFactory
		ok         bool
	}{
		{
			name: "meta built-in Codec encode v1.DeleteOptions",
			object: &metav1.DeleteOptions{
				GracePeriodSeconds: pointer.Int64Ptr(30),
				PropagationPolicy:  &policy,
			},
			gv:         metav1.SchemeGroupVersion,
			serializer: metainternalversionscheme.Codecs,
			ok:         true,
		},
		{
			name: "fake corev1 registered codec encode v1 DeleteOptions",
			object: &metav1.DeleteOptions{
				GracePeriodSeconds: pointer.Int64Ptr(30),
				PropagationPolicy:  &policy,
			},
			gv:         metav1.SchemeGroupVersion,
			serializer: testCodec,
			ok:         false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			codecs := &mockCodecs{}
			codecs.CodecFactory = test.serializer

			audit.LogRequestObject(ctx, deleteOption, test.gv, schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "pods",
			}, "", codecs)

			err := codecs.err
			if err != nil {
				if test.ok {
					t.Errorf("expect nil but got %#v", err)
				}
				t.Logf("encode object: %#v", err)
			} else {
				if !test.ok {
					t.Errorf("expect err but got nil")
				}
			}
		})
	}
}
