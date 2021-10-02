/*
Copyright 2020 The Kubernetes Authors.

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

package internal_test

import (
	"testing"
	"time"

	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
)

func TestAtMostEvery(t *testing.T) {
	duration := time.Second
	delay := 179 * time.Millisecond
	atMostEvery := internal.NewAtMostEvery(delay)
	count := 0
	exit := time.NewTicker(duration)
	tick := time.NewTicker(2 * time.Millisecond)
	defer exit.Stop()
	defer tick.Stop()

	done := false
	for !done {
		select {
		case <-exit.C:
			done = true
		case <-tick.C:
			atMostEvery.Do(func() {
				count++
			})
		}
	}

	if expected := int(duration/delay) + 1; count > expected {
		t.Fatalf("Function called %d times, should have been called less than or equal to %d times", count, expected)
	}
}
