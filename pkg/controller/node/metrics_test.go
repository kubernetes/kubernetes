/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestEvictionData(t *testing.T) {
	evictionData := newEvictionData(time.Hour)
	now := unversioned.Now()
	evictionData.now = func() unversioned.Time {
		return *(&now)
	}
	if evictionData.countEvictions("zone1") != 0 {
		t.Fatalf("Invalid eviction count before doing anything")
	}
	evictionData.initZone("zone1")
	if evictionData.countEvictions("zone1") != 0 {
		t.Fatalf("Invalid eviction after zone initialization")
	}

	evictionData.registerEviction("first", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 1 {
		t.Fatalf("Invalid eviction count after adding first Node")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.registerEviction("second", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 2 {
		t.Fatalf("Invalid eviction count after adding second Node")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.registerEviction("second", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 2 {
		t.Fatalf("Invalid eviction count after adding second Node second time")
	}
	if evictionData.countEvictions("zone2") != 0 {
		t.Fatalf("Invalid eviction in nonexistent zone")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.registerEviction("third", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 3 {
		t.Fatalf("Invalid eviction count after adding third Node first time")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.removeEviction("third", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 2 {
		t.Fatalf("Invalid eviction count after remove third Node")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.removeEviction("third", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 2 {
		t.Fatalf("Invalid eviction count after remove third Node second time")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.registerEviction("fourth", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 3 {
		t.Fatalf("Invalid eviction count after adding fourth Node first time")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.registerEviction("fourth", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 3 {
		t.Fatalf("Invalid eviction count after adding fourth Node second time")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.removeEviction("fourth", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 3 {
		t.Fatalf("Invalid eviction count after remove fourth Node first time")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.removeEviction("fourth", "zone1")
	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 2 {
		t.Fatalf("Invalid eviction count after remove fourth Node second time")
	}
	now = unversioned.NewTime(now.Add(52 * time.Minute))

	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 1 {
		t.Fatalf("Invalid eviction count after first Node went out of scope")
	}
	now = unversioned.NewTime(now.Add(time.Minute))

	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 1 {
		t.Fatalf("Invalid eviction count after first occurence of the second Node went out of scope")
	}
	now = unversioned.NewTime(now.Add(time.Second))

	evictionData.slideWindow()
	if evictionData.countEvictions("zone1") != 0 {
		t.Fatalf("Invalid eviction count after second occurence of the second Node went out of scope")
	}
}
