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

package request

import (
	"context"
	"net/http"

	"k8s.io/apimachinery/pkg/types"
)

type auditIDKeyType int

// auditIDKey is the key to associate the Audit-ID value of a request.
const auditIDKey auditIDKeyType = iota

// WithAuditID returns a copy of the parent context into which the Audit-ID
// associated with the request is set.
//
// If the specified auditID is empty, no value is set and the parent context is returned as is.
func WithAuditID(parent context.Context, auditID types.UID) context.Context {
	if auditID == "" {
		return parent
	}
	return WithValue(parent, auditIDKey, auditID)
}

// AuditIDFrom returns the value of the audit ID from the request context.
func AuditIDFrom(ctx context.Context) (types.UID, bool) {
	auditID, ok := ctx.Value(auditIDKey).(types.UID)
	return auditID, ok
}

// GetAuditIDTruncated returns the audit ID (truncated) associated with a request.
// If the length of the Audit-ID value exceeds the limit, we truncate it to keep
// the first N (maxAuditIDLength) characters.
// This is intended to be used in logging only.
func GetAuditIDTruncated(req *http.Request) string {
	auditID, ok := AuditIDFrom(req.Context())
	if !ok {
		return ""
	}

	// if the user has specified a very long audit ID then we will use the first N characters
	// Note: assuming Audit-ID header is in ASCII
	const maxAuditIDLength = 64
	if len(auditID) > maxAuditIDLength {
		auditID = auditID[0:maxAuditIDLength]
	}

	return string(auditID)
}
