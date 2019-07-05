/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package grpcrand implements math/rand functions in a concurrent-safe way
// with a global random source, independent of math/rand's global source.
package grpcrand

import (
	"math/rand"
	"sync"
	"time"
)

var (
	r  = rand.New(rand.NewSource(time.Now().UnixNano()))
	mu sync.Mutex
)

// Int63n implements rand.Int63n on the grpcrand global source.
func Int63n(n int64) int64 {
	mu.Lock()
	res := r.Int63n(n)
	mu.Unlock()
	return res
}

// Intn implements rand.Intn on the grpcrand global source.
func Intn(n int) int {
	mu.Lock()
	res := r.Intn(n)
	mu.Unlock()
	return res
}

// Float64 implements rand.Float64 on the grpcrand global source.
func Float64() float64 {
	mu.Lock()
	res := r.Float64()
	mu.Unlock()
	return res
}
