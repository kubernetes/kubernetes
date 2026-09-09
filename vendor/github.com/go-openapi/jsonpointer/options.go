// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonpointer

import (
	"sync"

	"github.com/go-openapi/jsonpointer/jsonname"
)

// Option to tune the behavior of a JSON [Pointer].
type Option func(*options)

var (
	//nolint:gochecknoglobals // package level defaults are provided as a convenient, backward-compatible way to adopt options.
	defaultOptions = options{
		provider: jsonname.DefaultJSONNameProvider,
	}
	//nolint:gochecknoglobals // guards defaultOptions against concurrent SetDefaultNameProvider / read races (testing)
	defaultOptionsMu sync.RWMutex
)

// SetDefaultNameProvider sets the [NameProvider] as a package-level default.
//
// By default, the default provider is [jsonname.DefaultJSONNameProvider].
//
// It is safe to call concurrently with [Pointer.Get], [Pointer.Set], [GetForToken] and
// [SetForToken].
// The typical usage is to call it once at initialization time.
//
// A nil provider is ignored.
func SetDefaultNameProvider(provider NameProvider) {
	if provider == nil {
		return
	}

	defaultOptionsMu.Lock()
	defer defaultOptionsMu.Unlock()

	defaultOptions.provider = provider
}

// UseGoNameProvider sets the [NameProvider] as a package-level default to the alternative provider
// [jsonname.GoNameProvider], that covers a few areas not supported by the default name provider.
//
// This implementation supports untagged exported fields and embedded types in go struct.
// It follows strictly the behavior of the JSON standard library regarding field naming conventions.
//
// It is safe to call concurrently with [Pointer.Get], [Pointer.Set], [GetForToken] and
// [SetForToken].
// The typical usage is to call it once at initialization time.
func UseGoNameProvider() {
	SetDefaultNameProvider(jsonname.NewGoNameProvider())
}

// DefaultNameProvider returns the current package-level [NameProvider].
func DefaultNameProvider() NameProvider { //nolint:ireturn // returning the interface is the point — callers pick their own implementation.
	defaultOptionsMu.RLock()
	defer defaultOptionsMu.RUnlock()

	return defaultOptions.provider
}

// WithNameProvider injects a custom [NameProvider] to resolve json names from go struct types.
func WithNameProvider(provider NameProvider) Option {
	return func(o *options) {
		o.provider = provider
	}
}

type options struct {
	provider NameProvider
}

func optionsWithDefaults(opts []Option) options {
	var o options
	o.provider = DefaultNameProvider()

	for _, apply := range opts {
		apply(&o)
	}

	return o
}
