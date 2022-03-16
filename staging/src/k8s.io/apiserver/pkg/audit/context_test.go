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
