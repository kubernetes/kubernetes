/*
Copyright 2020 The Kubernetes Authors.

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

	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// The key type is unexported to prevent collisions
type key int

const (
	// auditAnnotationsKey is the context key for the audit annotations.
	auditAnnotationsKey key = iota
)

// annotations = *[]annotation instead of a map to preserve order of insertions
type annotation struct {
	key, value string
}

// WithAuditAnnotations returns a new context that can store audit annotations
// via the AddAuditAnnotation function.  This function is meant to be called from
// an early request handler to allow all later layers to set audit annotations.
// This is required to support flows where handlers that come before WithAudit
// (such as WithAuthentication) wish to set audit annotations.
func WithAuditAnnotations(parent context.Context) context.Context {
	// this should never really happen, but prevent double registration of this slice
	if _, ok := parent.Value(auditAnnotationsKey).(*[]annotation); ok {
		return parent
	}

	var annotations []annotation // avoid allocations until we actually need it
	return genericapirequest.WithValue(parent, auditAnnotationsKey, &annotations)
}

// AddAuditAnnotation sets the audit annotation for the given key, value pair.
// It is safe to call at most parts of request flow that come after WithAuditAnnotations.
// The notable exception being that this function must not be called via a
// defer statement (i.e. after ServeHTTP) in a handler that runs before WithAudit
// as at that point the audit event has already been sent to the audit sink.
// Handlers that are unaware of their position in the overall request flow should
// prefer AddAuditAnnotation over LogAnnotation to avoid dropping annotations.
func AddAuditAnnotation(ctx context.Context, key, value string) {
	// use the audit event directly if we have it
	if ae := genericapirequest.AuditEventFrom(ctx); ae != nil {
		LogAnnotation(ae, key, value)
		return
	}

	annotations, ok := ctx.Value(auditAnnotationsKey).(*[]annotation)
	if !ok {
		return // adding audit annotation is not supported at this call site
	}

	*annotations = append(*annotations, annotation{key: key, value: value})
}

// This is private to prevent reads/write to the slice from outside of this package.
// The audit event should be directly read to get access to the annotations.
func auditAnnotationsFrom(ctx context.Context) []annotation {
	annotations, ok := ctx.Value(auditAnnotationsKey).(*[]annotation)
	if !ok {
		return nil // adding audit annotation is not supported at this call site
	}

	return *annotations
}
