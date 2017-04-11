// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metadata

import (
	"os"
	"sync"
	"testing"
)

func TestOnGCE_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	var last bool
	for i := 0; i < 100; i++ {
		onGCEOnce = sync.Once{}

		now := OnGCE()
		if i > 0 && now != last {
			t.Errorf("%d. changed from %v to %v", i, last, now)
		}
		last = now
	}
	t.Logf("OnGCE() = %v", last)
}

func TestOnGCE_Force(t *testing.T) {
	onGCEOnce = sync.Once{}
	old := os.Getenv(metadataHostEnv)
	defer os.Setenv(metadataHostEnv, old)
	os.Setenv(metadataHostEnv, "127.0.0.1")
	if !OnGCE() {
		t.Error("OnGCE() = false; want true")
	}
}
