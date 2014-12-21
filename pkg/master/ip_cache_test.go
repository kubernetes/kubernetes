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

package master

import (
	"testing"
	"time"

	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
)

type fakeClock struct {
	t time.Time
}

func (f *fakeClock) Now() time.Time {
	return f.t
}

func TestCacheExpire(t *testing.T) {
	fakeCloud := &fake_cloud.FakeCloud{}
	clock := &fakeClock{t: time.Now()}

	c := NewIPCache(fakeCloud, clock)

	_ = c.GetInstanceIP("foo")
	// This call should hit the cache, so we expect no additional calls to the cloud
	_ = c.GetInstanceIP("foo")
	// Advance the clock, this call should miss the cache, so expect one more call.
	clock.t = clock.t.Add(60 * time.Second)
	_ = c.GetInstanceIP("foo")

	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[1] != "ip-address" || fakeCloud.Calls[0] != "ip-address" {
		t.Errorf("Unexpected calls: %+v", fakeCloud.Calls)
	}
}
