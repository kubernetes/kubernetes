/*
Copyright 2023 The Kubernetes Authors.

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

package resource

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
)

// fallbackQueryParamVerifier encapsulates the primary Verifier that
// is invoked, and the secondary/fallback Verifier.
type fallbackQueryParamVerifier struct {
	primary   Verifier
	secondary Verifier
}

var _ Verifier = &fallbackQueryParamVerifier{}

// NewFallbackQueryParamVerifier returns a new Verifier which will invoke the
// initial/primary Verifier. If the primary Verifier is "NotFound", then the
// secondary Verifier is invoked as a fallback.
func NewFallbackQueryParamVerifier(primary Verifier, secondary Verifier) Verifier {
	return &fallbackQueryParamVerifier{
		primary:   primary,
		secondary: secondary,
	}
}

// HasSupport returns an error if the passed GVK does not support the
// query param (fieldValidation), as determined by the primary and
// secondary OpenAPI endpoints. The primary endoint is checked first,
// but if there is an error retrieving the OpenAPI V3 document, the
// secondary attempts to determine support. If the GVK supports the query param,
// nil is returned.
func (f *fallbackQueryParamVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	err := f.primary.HasSupport(gvk)
	// If an error was returned from the primary OpenAPI endpoint,
	// we fallback to check the secondary OpenAPI endpoint for
	// any error *except* "paramUnsupportedError".
	if err != nil && !IsParamUnsupportedError(err) {
		klog.V(7).Infof("openapi v3 error...falling back to legacy: %s", err)
		err = f.secondary.HasSupport(gvk)
	}
	return err
}
