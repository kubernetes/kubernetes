/*
Copyright 2023 The Kubernetes Authors.

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

package internal

import (
	"bytes"
	"testing"
)

const (
	testQPS = 1
	// testBurst should be no more than 9 since 9*100millisecond (test latency) = 900ms, which guarantees there is enough bursts per second.
	testBurst = 5
)

func TestMockRateLimitRemoteService(t *testing.T) {
	t.Parallel()
	ctx := testContext(t)
	plaintext := []byte(testPlaintext)
	aesService, err := NewMockAESService(testAESKey, testKeyID)
	if err != nil {
		t.Fatal(err)
	}
	mockLatencyService := NewMockLatencyService(aesService, testLatencyInMillisecond)
	kmsService := NewMockRateLimitService(mockLatencyService, testQPS, testBurst)

	t.Run("should hit rate limit", func(t *testing.T) {
		rateLimitExceeded := false
		for i := 0; i < 100; i++ {
			encRes, err := kmsService.Encrypt(ctx, "", plaintext)
			if i >= testBurst {
				if err != nil {
					if err.Error() != "rpc error: code = ResourceExhausted desc = remote encrypt rate limit exceeded" {
						t.Fatalf("should have failed with rate limit exceeded %d, have err: %v", testBurst, err)
					}
					rateLimitExceeded = true
				} else {
					if bytes.Equal(plaintext, encRes.Ciphertext) {
						t.Fatal("plaintext and ciphertext shouldn't be equal!")
					}
				}
			} else {
				if err != nil {
					t.Fatalf("err: %v, i: %d", err, i)
				}
				if bytes.Equal(plaintext, encRes.Ciphertext) {
					t.Fatal("plaintext and ciphertext shouldn't be equal!")
				}
			}
			// status should not hit any rate limit
			_, err = kmsService.Status(ctx)
			if err != nil {
				t.Fatal(err)
			}
		}
		if !rateLimitExceeded {
			t.Errorf("should have reached the rate limit of %d", testBurst)
		}
	})
}
