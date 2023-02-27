//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azcore

import (
	"context"
	"reflect"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
)

// AccessToken represents an Azure service bearer access token with expiry information.
type AccessToken struct {
	Token     string
	ExpiresOn time.Time
}

// TokenCredential represents a credential capable of providing an OAuth token.
type TokenCredential interface {
	// GetToken requests an access token for the specified set of scopes.
	GetToken(ctx context.Context, options policy.TokenRequestOptions) (AccessToken, error)
}

// holds sentinel values used to send nulls
var nullables map[reflect.Type]interface{} = map[reflect.Type]interface{}{}

// NullValue is used to send an explicit 'null' within a request.
// This is typically used in JSON-MERGE-PATCH operations to delete a value.
func NullValue[T any]() T {
	t := shared.TypeOfT[T]()
	v, found := nullables[t]
	if !found {
		var o reflect.Value
		if k := t.Kind(); k == reflect.Map {
			o = reflect.MakeMap(t)
		} else if k == reflect.Slice {
			// empty slices appear to all point to the same data block
			// which causes comparisons to become ambiguous.  so we create
			// a slice with len/cap of one which ensures a unique address.
			o = reflect.MakeSlice(t, 1, 1)
		} else {
			o = reflect.New(t.Elem())
		}
		v = o.Interface()
		nullables[t] = v
	}
	// return the sentinel object
	return v.(T)
}

// IsNullValue returns true if the field contains a null sentinel value.
// This is used by custom marshallers to properly encode a null value.
func IsNullValue[T any](v T) bool {
	// see if our map has a sentinel object for this *T
	t := reflect.TypeOf(v)
	if o, found := nullables[t]; found {
		o1 := reflect.ValueOf(o)
		v1 := reflect.ValueOf(v)
		// we found it; return true if v points to the sentinel object.
		// NOTE: maps and slices can only be compared to nil, else you get
		// a runtime panic.  so we compare addresses instead.
		return o1.Pointer() == v1.Pointer()
	}
	// no sentinel object for this *t
	return false
}

// ClientOptions contains configuration settings for a client's pipeline.
type ClientOptions = policy.ClientOptions
