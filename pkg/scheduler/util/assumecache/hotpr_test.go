/*
Copyright 2017 The Kubernetes Authors.

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

package assumecache

import (
	"testing"
)

func TestResolveConflict_TierBound(t *testing.T) {
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 5, 10, true)

	if result.Tier != TierBound {
		t.Errorf("expected TierBound, got %d", result.Tier)
	}
	if result.Resolution != "bound" {
		t.Errorf("expected resolution 'bound', got %q", result.Resolution)
	}
	if result.MergedObj != assumed {
		t.Errorf("expected assumed object to win when bound, got %v", result.MergedObj)
	}
	if result.AssumeGen != 5 {
		t.Errorf("expected AssumeGen=5, got %d", result.AssumeGen)
	}
	if result.EtcdGen != 10 {
		t.Errorf("expected EtcdGen=10, got %d", result.EtcdGen)
	}
}

func TestResolveConflict_TierBound_OverridesVersion(t *testing.T) {
	// Even when informer version is much higher, bound wins.
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 1, 100, true)

	if result.Tier != TierBound {
		t.Errorf("expected TierBound even with higher informer version, got %d", result.Tier)
	}
	if result.MergedObj != assumed {
		t.Errorf("expected assumed object when bound")
	}
}

func TestResolveConflict_TierAutoMerge_AssumeNewer(t *testing.T) {
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 10, 5, false)

	if result.Tier != TierAutoMerge {
		t.Errorf("expected TierAutoMerge, got %d", result.Tier)
	}
	if result.Resolution != "merged" {
		t.Errorf("expected resolution 'merged', got %q", result.Resolution)
	}
	if result.MergedObj != assumed {
		t.Errorf("expected assumed object to win when newer, got %v", result.MergedObj)
	}
}

func TestResolveConflict_TierAutoMerge_EqualVersion(t *testing.T) {
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 7, 7, false)

	if result.Tier != TierAutoMerge {
		t.Errorf("expected TierAutoMerge on equal versions, got %d", result.Tier)
	}
	if result.MergedObj != assumed {
		t.Errorf("expected assumed object on equal versions")
	}
}

func TestResolveConflict_TierEtcdWins(t *testing.T) {
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 5, 10, false)

	if result.Tier != TierEtcdWins {
		t.Errorf("expected TierEtcdWins, got %d", result.Tier)
	}
	if result.Resolution != "etcd_wins" {
		t.Errorf("expected resolution 'etcd_wins', got %q", result.Resolution)
	}
	if result.MergedObj != informer {
		t.Errorf("expected informer object to win when newer, got %v", result.MergedObj)
	}
	if result.AssumeGen != 5 {
		t.Errorf("expected AssumeGen=5, got %d", result.AssumeGen)
	}
	if result.EtcdGen != 10 {
		t.Errorf("expected EtcdGen=10, got %d", result.EtcdGen)
	}
}

func TestResolveConflict_TierEtcdWins_ByOne(t *testing.T) {
	assumed := "assumed-obj"
	informer := "informer-obj"

	result := ResolveConflict(assumed, informer, 9, 10, false)

	if result.Tier != TierEtcdWins {
		t.Errorf("expected TierEtcdWins when informer is one version ahead, got %d", result.Tier)
	}
	if result.MergedObj != informer {
		t.Errorf("expected informer object")
	}
}

func TestResolveConflict_NilObjects(t *testing.T) {
	// Ensure nil objects don't panic.
	result := ResolveConflict(nil, nil, 1, 2, false)
	if result.Tier != TierEtcdWins {
		t.Errorf("expected TierEtcdWins with nil objects, got %d", result.Tier)
	}
	if result.MergedObj != nil {
		t.Errorf("expected nil MergedObj when informer is nil")
	}
}
