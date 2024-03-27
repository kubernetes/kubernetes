/*
Copyright 2021 Portworx

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
package correlation

import (
	"context"

	"github.com/pborman/uuid"
	"google.golang.org/grpc/metadata"
)

// Component represents a control plane component for
// correlating requests
type Component string

// contextKeyType represents a key for interacting with the
// corrleation context request object
type contextKeyType string

const (
	// ContextKey represents the key for storing and retrieving
	// the correlation context in a context.Context object.
	ContextKey = contextKeyType("correlation-context")

	// ContextIDKey represents the key for the correlation ID
	ContextIDKey = "correlation-context-id"

	// ContextOriginKey represents the key for the correlation origin
	ContextOriginKey = "correlation-context-origin"

	ComponentUnknown   = Component("unknown")
	ComponentCSIDriver = Component("csi-driver")
	ComponentSDK       = Component("sdk-server")
	ComponentAuth      = Component("openstorage/pkg/auth")
)

// RequestContext represents the context for a given a request.
// A request represents a single action received from an SDK
// user, container orchestrator, or any other request.
type RequestContext struct {
	// ID is a randomly generated UUID per requst
	ID string

	// Origin is the starting point for this request.
	// Examples may include any of the following:
	// pxctl, pxc, kubernetes, CSI, SDK, etc
	Origin Component
}

// WithCorrelationContext returns a new correlation context object
func WithCorrelationContext(ctx context.Context, origin Component) context.Context {
	if v := ctx.Value(ContextKey); v == nil {
		requestContext := &RequestContext{
			ID:     uuid.New(),
			Origin: origin,
		}
		ctx = context.WithValue(ctx, ContextKey, requestContext)
	}

	return ctx
}

// TODO is an alias for context.TODO(), specifically
// for keeping track of areas where we might want to add
// the correlation context.
func TODO() context.Context {
	return context.TODO()
}

// RequestContextFromContextValue returns the request context from a context value
func RequestContextFromContextValue(ctx context.Context) *RequestContext {
	contextKeyValue := ctx.Value(ContextKey)
	rc, ok := contextKeyValue.(*RequestContext)
	if !ok {
		return &RequestContext{}
	}

	return rc
}

// RequestContextFromContextMetadata returns a new request context from a metadata object
func RequestContextFromContextMetadata(md metadata.MD) *RequestContext {
	rc := &RequestContext{}
	if len(md[ContextIDKey]) > 0 {
		rc.ID = md[ContextIDKey][0]
	}
	if len(md[ContextOriginKey]) > 0 {
		rc.Origin = Component(md[ContextOriginKey][0])
	}

	return rc
}

// AsMap returns the request context as a map
func (rc *RequestContext) AsMap() map[string]string {
	m := make(map[string]string)
	if rc.ID != "" {
		m[ContextIDKey] = rc.ID
	}
	if rc.Origin != "" {
		m[ContextOriginKey] = string(rc.Origin)
	}

	return m
}
