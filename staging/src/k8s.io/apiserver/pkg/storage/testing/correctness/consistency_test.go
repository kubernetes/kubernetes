/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage"
)

func TestGetReadConsistency(t *testing.T) {
	testCases := []struct {
		rv              string
		wantConsistency ReadConsistency
	}{
		{rv: "", wantConsistency: ConsistencyConsistent},
		{rv: "0", wantConsistency: ConsistencyNotOlderThan},
		{rv: "100", wantConsistency: ConsistencyNotOlderThan},
	}
	for i := range testCases {
		tc := &testCases[i]
		t.Run(fmt.Sprintf("rv=%q", tc.rv), func(t *testing.T) {
			got := GetReadConsistency(storage.GetOptions{ResourceVersion: tc.rv})
			require.Equal(t, tc.wantConsistency, got)
		})
	}
}

func TestListReadConsistency(t *testing.T) {
	continueTokenRV, err := storage.EncodeContinue("/pods/ns/p1", "/pods/", 100)
	require.NoError(t, err)
	continueTokenConsistent, err := storage.EncodeContinue("/pods/ns/p1", "/pods/", -1)
	require.NoError(t, err)

	// Table follows https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/2340-Consistent-reads-from-cache/README.md#how-to-debug-cache-issue
	testCases := []struct {
		rv              string
		match           metav1.ResourceVersionMatch
		continueToken   string
		limit           int64
		wantConsistency ReadConsistency
		wantErr         bool
	}{
		// ResourceVersion = unset ("")
		{rv: "", match: "", continueToken: "", limit: 0, wantConsistency: ConsistencyConsistent},
		{rv: "", match: "", continueToken: "", limit: 10, wantConsistency: ConsistencyConsistent},
		{rv: "", match: "", continueToken: continueTokenRV, limit: 0, wantConsistency: ConsistencyExact},
		{rv: "", match: "", continueToken: continueTokenRV, limit: 10, wantConsistency: ConsistencyExact},
		{rv: "", match: "", continueToken: continueTokenConsistent, limit: 0, wantConsistency: ConsistencyConsistent},
		{rv: "", match: "", continueToken: continueTokenConsistent, limit: 10, wantConsistency: ConsistencyConsistent},
		{rv: "", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 0, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 10, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 10, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 0, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 10, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 10, wantErr: true},

		// ResourceVersion = "0"
		{rv: "0", match: "", continueToken: "", limit: 0, wantConsistency: ConsistencyNotOlderThan},
		{rv: "0", match: "", continueToken: "", limit: 10, wantConsistency: ConsistencyNotOlderThan},
		// Note: KEP-2340 table says "Quorum read request" for (rv="0", match="", continue=token),
		// but storage.ValidateListOptions and watchCache.waitUntilFreshAndList actually use the
		// RV encoded in the continue token (Exact if continueRV > 0, Consistent if continueRV < 0).
		{rv: "0", match: "", continueToken: continueTokenRV, limit: 0, wantConsistency: ConsistencyExact},
		{rv: "0", match: "", continueToken: continueTokenRV, limit: 10, wantConsistency: ConsistencyExact},
		{rv: "0", match: "", continueToken: continueTokenConsistent, limit: 0, wantConsistency: ConsistencyConsistent},
		{rv: "0", match: "", continueToken: continueTokenConsistent, limit: 10, wantConsistency: ConsistencyConsistent},
		{rv: "0", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 0, wantErr: true},
		{rv: "0", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 10, wantErr: true},
		{rv: "0", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "0", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 10, wantErr: true},
		{rv: "0", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 0, wantConsistency: ConsistencyNotOlderThan},
		{rv: "0", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 10, wantConsistency: ConsistencyNotOlderThan},
		// Note: KEP-2340 table says "Read request from RV encoded in token" for (rv="0", match="NotOlderThan", continue=token),
		// but validation.ValidateListOptions forbids setting resourceVersionMatch when continue is provided.
		{rv: "0", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "0", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 10, wantErr: true},

		// ResourceVersion = "100"
		{rv: "100", match: "", continueToken: "", limit: 0, wantConsistency: ConsistencyNotOlderThan},
		{rv: "100", match: "", continueToken: "", limit: 10, wantConsistency: ConsistencyExact},
		// Note: KEP-2340 table says "Read request from RV encoded in token" for (rv=RV, match="", continue=token),
		// but storage.ValidateListOptions rejects specifying a non-zero resourceVersion when continue is provided.
		{rv: "100", match: "", continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "100", match: "", continueToken: continueTokenRV, limit: 10, wantErr: true},
		{rv: "100", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 0, wantConsistency: ConsistencyExact},
		{rv: "100", match: metav1.ResourceVersionMatchExact, continueToken: "", limit: 10, wantConsistency: ConsistencyExact},
		{rv: "100", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "100", match: metav1.ResourceVersionMatchExact, continueToken: continueTokenRV, limit: 10, wantErr: true},
		{rv: "100", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 0, wantConsistency: ConsistencyNotOlderThan},
		{rv: "100", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: "", limit: 10, wantConsistency: ConsistencyNotOlderThan},
		{rv: "100", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 0, wantErr: true},
		{rv: "100", match: metav1.ResourceVersionMatchNotOlderThan, continueToken: continueTokenRV, limit: 10, wantErr: true},
	}

	for i := range testCases {
		tc := &testCases[i]
		name := fmt.Sprintf("rv=%q/match=%q/continue=%v/limit=%d", tc.rv, tc.match, tc.continueToken != "", tc.limit)
		t.Run(name, func(t *testing.T) {
			opts := storage.ListOptions{
				ResourceVersion:      tc.rv,
				ResourceVersionMatch: tc.match,
				Recursive:            true,
				Predicate: storage.SelectionPredicate{
					Continue: tc.continueToken,
					Limit:    tc.limit,
				},
			}
			got, err := ListReadConsistency(opts)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantConsistency, got)
		})
	}
}
