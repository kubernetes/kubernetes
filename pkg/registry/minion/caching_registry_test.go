/*
Copyright 2014 Google Inc. All rights reserved.

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

package minion

import (
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

type fakeClock struct {
	now time.Time
}

func (f *fakeClock) Now() time.Time {
	return f.now
}

func TestCachingHit(t *testing.T) {
	fakeClock := fakeClock{
		now: time.Unix(0, 0),
	}
	fakeRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2"})
	expected := []string{"m1", "m2", "m3"}
	cache := CachingRegistry{
		delegate:   fakeRegistry,
		ttl:        1 * time.Second,
		clock:      &fakeClock,
		lastUpdate: fakeClock.Now().Unix(),
		minions:    expected,
	}
	list, err := cache.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, expected) {
		t.Errorf("expected: %v, got %v", expected, list)
	}
}

func TestCachingMiss(t *testing.T) {
	fakeClock := fakeClock{
		now: time.Unix(0, 0),
	}
	fakeRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2"})
	expected := []string{"m1", "m2", "m3"}
	cache := CachingRegistry{
		delegate:   fakeRegistry,
		ttl:        1 * time.Second,
		clock:      &fakeClock,
		lastUpdate: fakeClock.Now().Unix(),
		minions:    expected,
	}
	fakeClock.now = time.Unix(3, 0)
	list, err := cache.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, fakeRegistry.Minions) {
		t.Errorf("expected: %v, got %v", fakeRegistry.Minions, list)
	}
}

func TestCachingInsert(t *testing.T) {
	fakeClock := fakeClock{
		now: time.Unix(0, 0),
	}
	fakeRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2"})
	expected := []string{"m1", "m2", "m3"}
	cache := CachingRegistry{
		delegate:   fakeRegistry,
		ttl:        1 * time.Second,
		clock:      &fakeClock,
		lastUpdate: fakeClock.Now().Unix(),
		minions:    expected,
	}
	err := cache.Insert("foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	list, err := cache.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, fakeRegistry.Minions) {
		t.Errorf("expected: %v, got %v", fakeRegistry.Minions, list)
	}
}

func TestCachingDelete(t *testing.T) {
	fakeClock := fakeClock{
		now: time.Unix(0, 0),
	}
	fakeRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2"})
	expected := []string{"m1", "m2", "m3"}
	cache := CachingRegistry{
		delegate:   fakeRegistry,
		ttl:        1 * time.Second,
		clock:      &fakeClock,
		lastUpdate: fakeClock.Now().Unix(),
		minions:    expected,
	}
	err := cache.Delete("m2")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	list, err := cache.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, fakeRegistry.Minions) {
		t.Errorf("expected: %v, got %v", fakeRegistry.Minions, list)
	}
}
