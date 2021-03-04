/*
Copyright 2019 The Kubernetes Authors.

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

package x509

import (
	"crypto/x509"
	"fmt"

	"k8s.io/client-go/util/cert"
)

// StaticVerifierFn is a VerifyOptionFunc that always returns the same value.  This allows verify options that cannot change.
func StaticVerifierFn(opts x509.VerifyOptions) VerifyOptionFunc {
	return func() (x509.VerifyOptions, bool) {
		return opts, true
	}
}

// NewStaticVerifierFromFile creates a new verification func from a file.  It reads the content and then fails.
// It will return a nil function if you pass an empty CA file.
func NewStaticVerifierFromFile(clientCA string) (VerifyOptionFunc, error) {
	if len(clientCA) == 0 {
		return nil, nil
	}

	// Wrap with an x509 verifier
	var err error
	opts := DefaultVerifyOptions()
	opts.Roots, err = cert.NewPool(clientCA)
	if err != nil {
		return nil, fmt.Errorf("error loading certs from  %s: %v", clientCA, err)
	}

	return StaticVerifierFn(opts), nil
}

// StringSliceProvider is a way to get a string slice value.  It is heavily used for authentication headers among other places.
type StringSliceProvider interface {
	// Value returns the current string slice.  Callers should never mutate the returned value.
	Value() []string
}

// StringSliceProviderFunc is a function that matches the StringSliceProvider interface
type StringSliceProviderFunc func() []string

// Value returns the current string slice.  Callers should never mutate the returned value.
func (d StringSliceProviderFunc) Value() []string {
	return d()
}

// StaticStringSlice a StringSliceProvider that returns a fixed value
type StaticStringSlice []string

// Value returns the current string slice.  Callers should never mutate the returned value.
func (s StaticStringSlice) Value() []string {
	return s
}
