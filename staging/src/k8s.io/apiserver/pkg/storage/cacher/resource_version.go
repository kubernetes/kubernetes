/*
Copyright The Kubernetes Authors.

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

package cacher

import (
	"context"
	"time"

	"golang.org/x/sync/singleflight"
	"k8s.io/utils/clock"
)

const (
	// minResourceVersionRequestInterval bounds how frequently cacher queries storage
	// for the current resource version, allowing concurrent callers within the window to batch.
	minResourceVersionRequestInterval = 5 * time.Millisecond
)

type currentResourceVersionBatcher struct {
	readResourceVersion func(context.Context) (uint64, error)
	clock               clock.Clock

	lastRequestTime time.Time
	batchCalls      singleflight.Group
}

func newCurrentResourVersionBatcher(readResourceVersion func(context.Context) (uint64, error), c clock.Clock) *currentResourceVersionBatcher {
	return &currentResourceVersionBatcher{
		readResourceVersion: readResourceVersion,
		clock:               c,
	}
}

func (g *currentResourceVersionBatcher) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	ch := g.batchCalls.DoChan("", func() (any, error) {
		// Ensure calls are not executed more frequently than minResourceVersionRequestInterval.
		if !g.lastRequestTime.IsZero() {
			if waitDuration := minResourceVersionRequestInterval - g.clock.Since(g.lastRequestTime); waitDuration > 0 {
				g.clock.Sleep(waitDuration)
			}
		}
		g.lastRequestTime = g.clock.Now()

		// To maintain linearizability we need to reset batch before starting request as we cannot use result of request started before the function call.
		g.batchCalls.Forget("")
		return g.readResourceVersion(context.Background())
	})
	select {
	case res := <-ch:
		return res.Val.(uint64), res.Err
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}
