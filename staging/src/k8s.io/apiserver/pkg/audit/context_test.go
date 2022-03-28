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
	"fmt"
	"sync"
	"testing"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"

	"github.com/stretchr/testify/assert"
)

func TestAddAuditAnnotation(t *testing.T) {
	const (
		annotationKeyTemplate = "test-annotation-%d"
		annotationValue       = "test-annotation-value"
		numAnnotations        = 10
	)

	expectAnnotations := func(t *testing.T, annotations map[string]string) {
		assert.Len(t, annotations, numAnnotations)
	}
	noopValidator := func(_ *testing.T, _ context.Context) {}
	preEventValidator := func(t *testing.T, ctx context.Context) {
		ev := auditinternal.Event{
			Level: auditinternal.LevelMetadata,
		}
		addAuditAnnotationsFrom(ctx, &ev)
		expectAnnotations(t, ev.Annotations)
	}
	postEventValidator := func(t *testing.T, ctx context.Context) {
		ev := AuditEventFrom(ctx)
		expectAnnotations(t, ev.Annotations)
	}
	postEventEmptyValidator := func(t *testing.T, ctx context.Context) {
		ev := AuditEventFrom(ctx)
		assert.Empty(t, ev.Annotations)
	}

	tests := []struct {
		description string
		ctx         context.Context
		validator   func(t *testing.T, ctx context.Context)
	}{{
		description: "no audit",
		ctx:         context.Background(),
		validator:   noopValidator,
	}, {
		description: "no annotations context",
		ctx:         WithAuditContext(context.Background(), newAuditContext(auditinternal.LevelMetadata)),
		validator:   postEventValidator,
	}, {
		description: "no audit context",
		ctx:         WithAuditAnnotations(context.Background()),
		validator:   preEventValidator,
	}, {
		description: "both contexts metadata level",
		ctx:         WithAuditContext(WithAuditAnnotations(context.Background()), newAuditContext(auditinternal.LevelMetadata)),
		validator:   postEventValidator,
	}, {
		description: "both contexts none level",
		ctx:         WithAuditContext(WithAuditAnnotations(context.Background()), newAuditContext(auditinternal.LevelNone)),
		validator:   postEventEmptyValidator,
	}}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			var wg sync.WaitGroup
			wg.Add(numAnnotations)
			for i := 0; i < numAnnotations; i++ {
				go func(i int) {
					AddAuditAnnotation(test.ctx, fmt.Sprintf(annotationKeyTemplate, i), annotationValue)
					wg.Done()
				}(i)
			}
			wg.Wait()

			test.validator(t, test.ctx)
		})
	}
}

func newAuditContext(l auditinternal.Level) *AuditContext {
	return &AuditContext{
		Event: &auditinternal.Event{
			Level: l,
		},
	}
}
