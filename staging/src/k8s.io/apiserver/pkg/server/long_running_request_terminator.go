/*
Copyright 2022 The Kubernetes Authors.

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

package server

import (
	"context"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type longRunningRequestTerminator struct {
	tracker map[*context.Context]func()
	lock    sync.Mutex
}

func newLongRunningRequestTerminator() *longRunningRequestTerminator {
	return &longRunningRequestTerminator{
		tracker: make(map[*context.Context]func()),
	}
}

func (l *longRunningRequestTerminator) Attach(ctx context.Context) (context.Context, context.CancelFunc) {
	l.lock.Lock()
	defer l.lock.Unlock()
	newCtx, cancelFn := context.WithCancel(ctx)

	newCancelFn := func() {
		l.lock.Lock()
		defer l.lock.Unlock()
		defer cancelFn()
		delete(l.tracker, &newCtx)
	}
	l.tracker[&newCtx] = newCancelFn
	return newCtx, newCancelFn
}

func (l *longRunningRequestTerminator) startTerminating(delay time.Duration) {
	l.lock.Lock()
	defer l.lock.Unlock()

	fns := make([]func(), 0, len(l.tracker))
	for _, element := range l.tracker {
		fns = append(fns, element)
	}

	qps := len(fns) / int(delay.Seconds())
	limiter := rate.NewLimiter(rate.Limit(qps), 5)

	go func() {
		background := context.Background()
		for _, cancelFn := range fns {
			limiter.Wait(background)
			cancelFn()
		}
	}()
}
