// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sources

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/heapster/sinks/cache"
	kube_api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
)

func TestEventsBasic(t *testing.T) {
	t.Skip("Broken and flaky test - first fix #537 and #565")
	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "something",
		ResponseBody: body(&kube_api.EventList{}),
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Default.Version()})
	cache := cache.NewCache(time.Hour, time.Hour)
	source := NewKubeEvents(client, cache)
	_, err := source.GetInfo(time.Now(), time.Now().Add(time.Minute))
	require.NoError(t, err)
	require.NotEmpty(t, source.DebugInfo())
}
