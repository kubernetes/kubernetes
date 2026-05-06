/*
Copyright 2015 The Kubernetes Authors.

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

package podautoscaler

import (
	"testing"
	"time"
)

func TestPerItemIntervalRateLimiter_DefaultInterval(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)

	if got := rl.When("item-a"); got != 10*time.Second {
		t.Errorf("When(item-a) = %v, want default %v", got, 10*time.Second)
	}
	if got := rl.When("item-b"); got != 10*time.Second {
		t.Errorf("When(item-b) = %v, want default %v", got, 10*time.Second)
	}
}

func TestPerItemIntervalRateLimiter_PerItemOverride(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)
	rl.SetItemInterval("fast-item", 2*time.Second)
	rl.SetItemInterval("slow-item", 30*time.Second)

	if got := rl.When("fast-item"); got != 2*time.Second {
		t.Errorf("When(fast-item) = %v, want %v", got, 2*time.Second)
	}
	if got := rl.When("slow-item"); got != 30*time.Second {
		t.Errorf("When(slow-item) = %v, want %v", got, 30*time.Second)
	}
	if got := rl.When("default-item"); got != 10*time.Second {
		t.Errorf("When(default-item) = %v, want default %v", got, 10*time.Second)
	}
}

func TestPerItemIntervalRateLimiter_RemoveItem(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)
	rl.SetItemInterval("item", 3*time.Second)

	if got := rl.When("item"); got != 3*time.Second {
		t.Errorf("When(item) before remove = %v, want %v", got, 3*time.Second)
	}

	rl.RemoveItem("item")

	if got := rl.When("item"); got != 10*time.Second {
		t.Errorf("When(item) after remove = %v, want default %v", got, 10*time.Second)
	}
}

func TestPerItemIntervalRateLimiter_RemoveNonExistent(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)
	rl.RemoveItem("nonexistent") // should not panic
	if got := rl.When("nonexistent"); got != 10*time.Second {
		t.Errorf("When(nonexistent) = %v, want default %v", got, 10*time.Second)
	}
}

func TestPerItemIntervalRateLimiter_NumRequeues(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)
	if got := rl.NumRequeues("item"); got != 1 {
		t.Errorf("NumRequeues() = %d, want 1", got)
	}
}

func TestPerItemIntervalRateLimiter_Forget(t *testing.T) {
	rl := NewPerItemIntervalRateLimiter(10 * time.Second)
	rl.Forget("item") // should not panic
}

func TestNewDefaultHPARateLimiter(t *testing.T) {
	rl := NewDefaultHPARateLimiter(15 * time.Second)
	if rl == nil {
		t.Fatal("NewDefaultHPARateLimiter returned nil")
	}
	if got := rl.When("any"); got != 15*time.Second {
		t.Errorf("default HPA rate limiter When() = %v, want %v", got, 15*time.Second)
	}
}
