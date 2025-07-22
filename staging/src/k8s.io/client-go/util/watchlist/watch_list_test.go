/*
Copyright 2024 The Kubernetes Authors.

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

package watchlist

import (
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/utils/ptr"
)

// TestPrepareWatchListOptionsFromListOptions test the following cases:
//
// +--------------------------+-----------------+---------+-----------------+
// |   ResourceVersionMatch   | ResourceVersion |  Limit  |  Continuation   |
// +--------------------------+-----------------+---------+-----------------+
// | unset/NotOlderThan/Exact | unset/0/100     | unset/4 | unset/FakeToken |
// +--------------------------+-----------------+---------+-----------------+
func TestPrepareWatchListOptionsFromListOptions(t *testing.T) {
	scenarios := []struct {
		name              string
		listOptions       metav1.ListOptions
		enableWatchListFG bool

		expectToPrepareWatchListOptions bool
		expectedWatchListOptions        metav1.ListOptions
	}{

		{
			name:                            "can't enable watch list for: WatchListClient=off, RVM=unset, RV=unset, Limit=unset, Continuation=unset",
			enableWatchListFG:               false,
			expectToPrepareWatchListOptions: false,
		},
		//		+----------------------+-----------------+-------+--------------+
		//		| ResourceVersionMatch | ResourceVersion | Limit | Continuation |
		//		+----------------------+-----------------+-------+--------------+
		//		| unset                | unset           | unset | unset        |
		//		| unset                | 0               | unset | unset        |
		//		| unset                | 100             | unset | unset        |
		//		| unset                | 0               | 4     | unset        |
		//		| unset                | 0               | unset | FakeToken    |
		//		+----------------------+-----------------+-------+--------------+
		{
			name:                            "can enable watch list for: RVM=unset, RV=unset, Limit=unset, Continuation=unset",
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor(""),
		},
		{
			name:                            "can enable watch list for: RVM=unset, RV=0, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersion: "0"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("0"),
		},
		{
			name:                            "can enable watch list for: RVM=unset, RV=100, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersion: "100"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("100"),
		},
		{
			name:                            "legacy: can enable watch list for: RVM=unset, RV=0, Limit=4,  Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersion: "0", Limit: 4},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("0"),
		},
		{
			name:                            "can't enable watch list for: RVM=unset, RV=0, Limit=unset, Continuation=FakeToken",
			listOptions:                     metav1.ListOptions{ResourceVersion: "0", Continue: "FakeToken"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		//		+----------------------+-----------------+-------+--------------+
		//		| ResourceVersionMatch | ResourceVersion | Limit | Continuation |
		//		+----------------------+-----------------+-------+--------------+
		//		| NotOlderThan         | unset           | unset | unset        |
		//		| NotOlderThan         | 0               | unset | unset        |
		//		| NotOlderThan         | 100             | unset | unset        |
		//		| NotOlderThan         | 0               | 4     | unset        |
		//		| NotOlderThan         | 0               | unset | FakeToken    |
		//		+----------------------+-----------------+-------+--------------+
		{
			name:                            "can't enable watch list for: RVM=NotOlderThan, RV=unset, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		{
			name:                            "can enable watch list for: RVM=NotOlderThan, RV=0, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, ResourceVersion: "0"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("0"),
		},
		{
			name:                            "can enable watch list for: RVM=NotOlderThan, RV=100, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, ResourceVersion: "100"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("100"),
		},
		{
			name:                            "legacy: can enable watch list for: RVM=NotOlderThan, RV=0, Limit=4, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, ResourceVersion: "0", Limit: 4},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: true,
			expectedWatchListOptions:        expectedWatchListOptionsFor("0"),
		},
		{
			name:                            "can't enable watch list for: RVM=NotOlderThan, RV=0, Limit=unset, Continuation=FakeToken",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, ResourceVersion: "0", Continue: "FakeToken"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},

		//		+----------------------+-----------------+-------+--------------+
		//		| ResourceVersionMatch | ResourceVersion | Limit | Continuation |
		//		+----------------------+-----------------+-------+--------------+
		//		| Exact                | unset           | unset | unset        |
		//		| Exact                | 0               | unset | unset        |
		//		| Exact                | 100             | unset | unset        |
		//		| Exact                | 0               | 4     | unset        |
		//		| Exact                | 0               | unset | FakeToken    |
		//		+----------------------+-----------------+-------+--------------+
		{
			name:                            "can't enable watch list for: RVM=Exact, RV=unset, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchExact},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		{
			name:                            "can enable watch list for: RVM=Exact, RV=0, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchExact, ResourceVersion: "0"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		{
			name:                            "can enable watch list for: RVM=Exact, RV=100, Limit=unset, Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchExact, ResourceVersion: "100"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		{
			name:                            "can't enable watch list for: RVM=Exact, RV=0, Limit=4,  Continuation=unset",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchExact, ResourceVersion: "0", Limit: 4},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
		{
			name:                            "can't enable watch list for: RVM=Exact, RV=0, Limit=unset, Continuation=FakeToken",
			listOptions:                     metav1.ListOptions{ResourceVersionMatch: metav1.ResourceVersionMatchExact, ResourceVersion: "0", Continue: "FakeToken"},
			enableWatchListFG:               true,
			expectToPrepareWatchListOptions: false,
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, scenario.enableWatchListFG)

			watchListOptions, hasWatchListOptionsPrepared, err := PrepareWatchListOptionsFromListOptions(scenario.listOptions)

			require.NoError(t, err)
			require.Equal(t, scenario.expectToPrepareWatchListOptions, hasWatchListOptionsPrepared)
			require.Equal(t, scenario.expectedWatchListOptions, watchListOptions)
		})
	}
}

func expectedWatchListOptionsFor(rv string) metav1.ListOptions {
	var watchListOptions metav1.ListOptions

	watchListOptions.ResourceVersion = rv
	watchListOptions.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	watchListOptions.Watch = true
	watchListOptions.AllowWatchBookmarks = true
	watchListOptions.SendInitialEvents = ptr.To(true)

	return watchListOptions
}
