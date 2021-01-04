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

package controller

import (
	"context"
	"time"
)

const (
	// DefaultResyncPeriod is the default duration that is used when no
	// resync period is associated with a controllers initialization context.
	DefaultResyncPeriod = 10 * time.Hour
)

// Informer is the group of methods that a type must implement to be passed to
// StartInformers.
type Informer interface {
	Run(<-chan struct{})
	HasSynced() bool
}

// This is attached to contexts passed to controller constructors to associate
// a resync period.
type resyncPeriodKey struct{}

// WithResyncPeriod associates the given resync period with the given context in
// the context that is returned.
func WithResyncPeriod(ctx context.Context, resync time.Duration) context.Context {
	return context.WithValue(ctx, resyncPeriodKey{}, resync)
}

// GetResyncPeriod returns the resync period associated with the given context.
// When none is specified a default resync period is used.
func GetResyncPeriod(ctx context.Context) time.Duration {
	rp := ctx.Value(resyncPeriodKey{})
	if rp == nil {
		return DefaultResyncPeriod
	}
	return rp.(time.Duration)
}
