/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

func TestLogAnnotation(t *testing.T) {
	ev := &auditinternal.Event{
		Level:   auditinternal.LevelMetadata,
		AuditID: "fake id",
	}
	logAnnotation(ev, "foo", "bar")
	logAnnotation(ev, "foo", "baz")
	assert.Equal(t, "bar", ev.Annotations["foo"], "audit annotation should not be overwritten.")

	logAnnotation(ev, "qux", "")
	logAnnotation(ev, "qux", "baz")
	assert.Equal(t, "", ev.Annotations["qux"], "audit annotation should not be overwritten.")
}

func TestAddAuditAnnotations(t *testing.T) {
	ctxIndividual := WithAuditAnnotations(context.Background())
	ctxBulk := WithAuditAnnotations(context.Background())

	t.Log("Adding annotations before the event has been registered")
	var bulkAnnotations []string
	expectedAnnotations := map[string]string{}
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("pre-event-annotation-%d", i)
		value := "pre-event-annotation-value"

		expectedAnnotations[key] = value
		bulkAnnotations = append(bulkAnnotations, key, value)
		AddAuditAnnotation(ctxIndividual, key, value)
	}
	AddAuditAnnotations(ctxBulk, bulkAnnotations...)

	evIndividual := auditinternal.Event{Level: auditinternal.LevelMetadata}
	addAuditAnnotationsFrom(ctxIndividual, &evIndividual)

	evBulk := auditinternal.Event{Level: auditinternal.LevelMetadata}
	addAuditAnnotationsFrom(ctxBulk, &evBulk)

	assert.Equal(t, expectedAnnotations, evIndividual.Annotations, "pre-event individual calls")
	assert.Equal(t, expectedAnnotations, evBulk.Annotations, "pre-event bulk call")

	ctxIndividual = WithAuditContext(ctxIndividual, &AuditContext{Event: &evIndividual})
	ctxBulk = WithAuditContext(ctxBulk, &AuditContext{Event: &evBulk})

	t.Log("Adding annotations after the event has been registered")
	bulkAnnotations = []string{}
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("post-event-annotation-%d", i)
		value := "post-event-annotation-value"

		expectedAnnotations[key] = value
		bulkAnnotations = append(bulkAnnotations, key, value)
		AddAuditAnnotation(ctxIndividual, key, value)
	}
	AddAuditAnnotations(ctxBulk, bulkAnnotations...)

	assert.Equal(t, expectedAnnotations, GetAuditEventCopy(ctxIndividual).Annotations, "post-event individual calls")
	assert.Equal(t, expectedAnnotations, GetAuditEventCopy(ctxBulk).Annotations, "post-event bulk call")
}
