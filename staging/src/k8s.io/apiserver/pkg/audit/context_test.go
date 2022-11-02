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
	eventValidator := func(t *testing.T, ctx context.Context) {
		ev := AuditContextFrom(ctx).Event
		expectAnnotations(t, ev.Annotations)
	}
	eventEmptyValidator := func(t *testing.T, ctx context.Context) {
		ev := AuditContextFrom(ctx).Event
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
		description: "with metadata level",
		ctx:         withAuditContextAndLevel(context.Background(), auditinternal.LevelMetadata),
		validator:   eventValidator,
	}, {
		description: "with none level",
		ctx:         withAuditContextAndLevel(context.Background(), auditinternal.LevelNone),
		validator:   eventEmptyValidator,
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

func withAuditContextAndLevel(ctx context.Context, l auditinternal.Level) context.Context {
	ctx = WithAuditContext(ctx)
	ac := AuditContextFrom(ctx)
	ac.RequestAuditConfig.Level = l
	return ctx
}
