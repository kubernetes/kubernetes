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
	"sync"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

// The key type is unexported to prevent collisions
type key int

const (
	// auditAnnotationsKey is the context key for the audit annotations.
	// TODO: consolidate all audit info under the AuditContext, rather than storing 3 separate keys.
	auditAnnotationsKey key = iota

	// auditKey is the context key for storing the audit event that is being
	// captured and the evaluated policy that applies to the given request.
	auditKey

	// auditAnnotationsMutexKey is the context key for the audit annotations mutex.
	auditAnnotationsMutexKey
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
	parent = withAuditAnnotationsMutex(parent)

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
	mutex, ok := auditAnnotationsMutex(ctx)
	if !ok {
		// auditing is not enabled
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	ae := AuditEventFrom(ctx)
	var ctxAnnotations *[]annotation
	if ae == nil {
		ctxAnnotations, _ = ctx.Value(auditAnnotationsKey).(*[]annotation)
	}

	addAuditAnnotationLocked(ae, ctxAnnotations, key, value)
}

// AddAuditAnnotations is a bulk version of AddAuditAnnotation. Refer to AddAuditAnnotation for
// restrictions on when this can be called.
// keysAndValues are the key-value pairs to add, and must have an even number of items.
func AddAuditAnnotations(ctx context.Context, keysAndValues ...string) {
	mutex, ok := auditAnnotationsMutex(ctx)
	if !ok {
		// auditing is not enabled
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	ae := AuditEventFrom(ctx)
	var ctxAnnotations *[]annotation
	if ae == nil {
		ctxAnnotations, _ = ctx.Value(auditAnnotationsKey).(*[]annotation)
	}

	if len(keysAndValues)%2 != 0 {
		klog.Errorf("Dropping mismatched audit annotation %q", keysAndValues[len(keysAndValues)-1])
	}
	for i := 0; i < len(keysAndValues); i += 2 {
		addAuditAnnotationLocked(ae, ctxAnnotations, keysAndValues[i], keysAndValues[i+1])
	}
}

// AddAuditAnnotationsMap is a bulk version of AddAuditAnnotation. Refer to AddAuditAnnotation for
// restrictions on when this can be called.
func AddAuditAnnotationsMap(ctx context.Context, annotations map[string]string) {
	mutex, ok := auditAnnotationsMutex(ctx)
	if !ok {
		// auditing is not enabled
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	ae := AuditEventFrom(ctx)
	var ctxAnnotations *[]annotation
	if ae == nil {
		ctxAnnotations, _ = ctx.Value(auditAnnotationsKey).(*[]annotation)
	}

	for k, v := range annotations {
		addAuditAnnotationLocked(ae, ctxAnnotations, k, v)
	}
}

// addAuditAnnotationLocked is the shared code for recording an audit annotation. This method should
// only be called while the auditAnnotationsMutex is locked.
func addAuditAnnotationLocked(ae *auditinternal.Event, annotations *[]annotation, key, value string) {
	if ae != nil {
		logAnnotation(ae, key, value)
	} else if annotations != nil {
		*annotations = append(*annotations, annotation{key: key, value: value})
	}
}

// This is private to prevent reads/write to the slice from outside of this package.
// The audit event should be directly read to get access to the annotations.
func addAuditAnnotationsFrom(ctx context.Context, ev *auditinternal.Event) {
	mutex, ok := auditAnnotationsMutex(ctx)
	if !ok {
		// auditing is not enabled
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	annotations, ok := ctx.Value(auditAnnotationsKey).(*[]annotation)
	if !ok {
		return // no annotations to copy
	}

	for _, kv := range *annotations {
		logAnnotation(ev, kv.key, kv.value)
	}
}

// LogAnnotation fills in the Annotations according to the key value pair.
func logAnnotation(ae *auditinternal.Event, key, value string) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}
	if ae.Annotations == nil {
		ae.Annotations = make(map[string]string)
	}
	if v, ok := ae.Annotations[key]; ok && v != value {
		klog.Warningf("Failed to set annotations[%q] to %q for audit:%q, it has already been set to %q", key, value, ae.AuditID, ae.Annotations[key])
		return
	}
	ae.Annotations[key] = value
}

// WithAuditContext returns a new context that stores the pair of the audit
// configuration object that applies to the given request and
// the audit event that is going to be written to the API audit log.
func WithAuditContext(parent context.Context, ev *AuditContext) context.Context {
	parent = withAuditAnnotationsMutex(parent)
	return genericapirequest.WithValue(parent, auditKey, ev)
}

// AuditEventFrom returns the audit event struct on the ctx
func AuditEventFrom(ctx context.Context) *auditinternal.Event {
	if o := AuditContextFrom(ctx); o != nil {
		return o.Event
	}
	return nil
}

// AuditContextFrom returns the pair of the audit configuration object
// that applies to the given request and the audit event that is going to
// be written to the API audit log.
func AuditContextFrom(ctx context.Context) *AuditContext {
	ev, _ := ctx.Value(auditKey).(*AuditContext)
	return ev
}

// WithAuditAnnotationMutex adds a mutex for guarding context.AddAuditAnnotation.
func withAuditAnnotationsMutex(parent context.Context) context.Context {
	if _, ok := parent.Value(auditAnnotationsMutexKey).(*sync.Mutex); ok {
		return parent
	}
	var mutex sync.Mutex
	return genericapirequest.WithValue(parent, auditAnnotationsMutexKey, &mutex)
}

// AuditAnnotationsMutex returns the audit annotations mutex from the context.
func auditAnnotationsMutex(ctx context.Context) (*sync.Mutex, bool) {
	mutex, ok := ctx.Value(auditAnnotationsMutexKey).(*sync.Mutex)
	return mutex, ok
}
