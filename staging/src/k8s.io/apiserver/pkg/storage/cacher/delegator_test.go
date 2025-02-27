/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"
)

func TestCalculateDigest(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	// NewCacheDelegator(cacher, backingStorage)

	// testCases := []struct {
	// 	desc         string
	// 	expectDigest metrics.StorageDigest
	// }{
	// 	{
	// 		desc:         "",
	// 		expectDigest: metrics.StorageDigest{},
	// 	},
	// }

	// for _, tc := range testCases {
	// 	t.Run(tc.desc, func(t *testing.T) {
	// 		digest, err := delegator.consistency.calculateDigests(context.Background())
	// 		if err != nil {
	// 			t.Fatal(err)
	// 		}
	// 		if *digest != tc.expectDigest {
	// 			t.Errorf("Got: %+v, expected: %+v", *digest, tc.expectDigest)
	// 		}
	// 	})
	// }
}
