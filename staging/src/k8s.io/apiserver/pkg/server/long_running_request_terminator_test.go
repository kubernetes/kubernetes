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
	"testing"
	"time"
)

func TestLongRunningRequestTerminator(t *testing.T) {
	terminator := newLongRunningRequestTerminator()
	contexts := []context.Context{}

	bg := context.TODO()
	for i := 0; i < 10; i++ {
		newCtx, _ := terminator.Attach(bg)
		contexts = append(contexts, newCtx)
	}

	terminator.startTerminating(time.Second * 1)

	done := 0
	for _, ctx := range contexts {
		select {
		case <-ctx.Done():
			done += 1
		case <-time.After(time.Second * 2):
		}
	}
	if done != 10 {
		t.Fatalf("Want %d contexts to be cancelled, got %d", 10, done)
	}
}
