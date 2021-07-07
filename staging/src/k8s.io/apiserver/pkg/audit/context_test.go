/*
Copyright 2021 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithAuditAnnotations(t *testing.T) {
	var annotations []annotation
	expContext := context.WithValue(context.Background(), auditAnnotationsKey, &annotations)

	// Add key auditAnnotations to context
	gotContext := WithAuditAnnotations(context.Background())
	assert.Equal(t, expContext, gotContext)

	// Key auditAnnotations is already in context
	gotContext = WithAuditAnnotations(gotContext)
	assert.Equal(t, expContext, gotContext)
}

func TestAddAuditAnnotation(t *testing.T) {
	testCases := []struct {
		desc                string
		expEventAnnotations map[string]string
		expCtxAnnotations   []annotation
		getFunc             func(t *testing.T) (*audit.Event, []annotation)
	}{
		{
			desc:                "auditEvent in context",
			expEventAnnotations: map[string]string{"foo": "bar"},
			expCtxAnnotations:   ([]annotation)(nil),
			getFunc: func(t *testing.T) (*audit.Event, []annotation) {
				ev := &audit.Event{Level: audit.LevelMetadata}
				// add auditEvent to context,
				ctx := request.WithAuditEvent(context.Background(), ev)

				// record annotations to auditEvent not context
				AddAuditAnnotation(ctx, "foo", "bar")

				//get annotations from auditEvent and context
				gotCtxAnnotations := auditAnnotationsFrom(ctx)
				gotEvent := request.AuditEventFrom(ctx)
				return gotEvent, gotCtxAnnotations
			},
		},
		{
			desc:              "auditEvent and key auditAnnotations not in context",
			expCtxAnnotations: ([]annotation)(nil),
			getFunc: func(t *testing.T) (*audit.Event, []annotation) {
				ctx := context.Background()
				AddAuditAnnotation(ctx, "foo", "bar")
				gotCtxAnnotations := auditAnnotationsFrom(ctx)
				gotEvent := request.AuditEventFrom(ctx)
				return gotEvent, gotCtxAnnotations
			},
		},
		{
			desc:              "auditEvent not in context, key auditAnnotations in context",
			expCtxAnnotations: []annotation{{key: "foo", value: "bar"}},
			getFunc: func(t *testing.T) (*audit.Event, []annotation) {
				ctx := WithAuditAnnotations(context.Background())
				AddAuditAnnotation(ctx, "foo", "bar")
				gotCtxAnnotations := auditAnnotationsFrom(ctx)
				gotEvent := request.AuditEventFrom(ctx)
				return gotEvent, gotCtxAnnotations
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotEvent, gotCtxAnnotations := tc.getFunc(t)
			assert.Equal(t, tc.expCtxAnnotations, gotCtxAnnotations)
			if gotEvent != nil {
				assert.Equal(t, tc.expEventAnnotations, gotEvent.Annotations)
			}
		})
	}
}
