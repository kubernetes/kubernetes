/*
Copyright 2020 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateListOptions(t *testing.T) {
	boolPtrFn := func(b bool) *bool {
		return &b
	}

	cases := []struct {
		name                    string
		opts                    internalversion.ListOptions
		watchListFeatureEnabled bool
		expectErrors            []string
	}{{
		name: "valid-default",
		opts: internalversion.ListOptions{},
	}, {
		name: "valid-resourceversionmatch-exact",
		opts: internalversion.ListOptions{
			ResourceVersion:      "1",
			ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		},
	}, {
		name: "invalid-resourceversionmatch-exact",
		opts: internalversion.ListOptions{
			ResourceVersion:      "0",
			ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		},
		expectErrors: []string{"resourceVersionMatch: Forbidden: resourceVersionMatch \"exact\" is forbidden for resourceVersion \"0\""},
	}, {
		name: "valid-resourceversionmatch-notolderthan",
		opts: internalversion.ListOptions{
			ResourceVersion:      "0",
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		},
	}, {
		name: "invalid-resourceversionmatch",
		opts: internalversion.ListOptions{
			ResourceVersion:      "0",
			ResourceVersionMatch: "foo",
		},
		expectErrors: []string{"resourceVersionMatch: Unsupported value: \"foo\": supported values: \"Exact\", \"NotOlderThan\", \"\""},
	}, {
		name: "list-sendInitialEvents-forbidden",
		opts: internalversion.ListOptions{
			SendInitialEvents: boolPtrFn(true),
		},
		expectErrors: []string{"sendInitialEvents: Forbidden: sendInitialEvents is forbidden for list"},
	}, {
		name: "valid-watch-default",
		opts: internalversion.ListOptions{
			Watch: true,
		},
	}, {
		name: "valid-watch-sendInitialEvents-on",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			AllowWatchBookmarks:  true,
		},
		watchListFeatureEnabled: true,
	}, {
		name: "valid-watch-sendInitialEvents-off",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(false),
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			AllowWatchBookmarks:  true,
		},
		watchListFeatureEnabled: true,
	}, {
		name: "watch-resourceversionmatch-without-sendInitialEvents-forbidden",
		opts: internalversion.ListOptions{
			Watch:                true,
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		},
		expectErrors: []string{"resourceVersionMatch: Forbidden: resourceVersionMatch is forbidden for watch unless sendInitialEvents is provided"},
	}, {
		name: "watch-sendInitialEvents-without-resourceversionmatch-forbidden",
		opts: internalversion.ListOptions{
			Watch:             true,
			SendInitialEvents: boolPtrFn(true),
		},
		expectErrors: []string{"resourceVersionMatch: Forbidden: sendInitialEvents requires setting resourceVersionMatch to NotOlderThan", "sendInitialEvents: Forbidden: sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled"},
	}, {
		name: "watch-sendInitialEvents-with-exact-resourceversionmatch-forbidden",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			AllowWatchBookmarks:  true,
		},
		watchListFeatureEnabled: true,
		expectErrors:            []string{"resourceVersionMatch: Forbidden: sendInitialEvents requires setting resourceVersionMatch to NotOlderThan", "resourceVersionMatch: Unsupported value: \"Exact\": supported values: \"NotOlderThan\""},
	}, {
		name: "watch-sendInitialEvents-on-with-empty-resourceversionmatch-forbidden",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: "",
		},
		expectErrors: []string{"resourceVersionMatch: Forbidden: sendInitialEvents requires setting resourceVersionMatch to NotOlderThan", "sendInitialEvents: Forbidden: sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled"},
	}, {
		name: "watch-sendInitialEvents-off-with-empty-resourceversionmatch-forbidden",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(false),
			ResourceVersionMatch: "",
		},
		expectErrors: []string{"resourceVersionMatch: Forbidden: sendInitialEvents requires setting resourceVersionMatch to NotOlderThan", "sendInitialEvents: Forbidden: sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled"},
	}, {
		name: "watch-sendInitialEvents-with-incorrect-resourceversionmatch-forbidden",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: "incorrect",
			AllowWatchBookmarks:  true,
		},
		watchListFeatureEnabled: true,
		expectErrors:            []string{"resourceVersionMatch: Forbidden: sendInitialEvents requires setting resourceVersionMatch to NotOlderThan", "resourceVersionMatch: Unsupported value: \"incorrect\": supported values: \"NotOlderThan\""},
	}, {
		// note that validating allowWatchBookmarks would break backward compatibility
		// because it was possible to request initial events via resourceVersion=0 before this change
		name: "watch-sendInitialEvents-no-allowWatchBookmark",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		},
		watchListFeatureEnabled: true,
	}, {
		name: "watch-sendInitialEvents-no-watchlist-fg-disabled",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			AllowWatchBookmarks:  true,
		},
		expectErrors: []string{"sendInitialEvents: Forbidden: sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled"},
	}, {
		name: "watch-sendInitialEvents-no-watchlist-fg-disabled",
		opts: internalversion.ListOptions{
			Watch:                true,
			SendInitialEvents:    boolPtrFn(true),
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			AllowWatchBookmarks:  true,
			Continue:             "123",
		},
		watchListFeatureEnabled: true,
		expectErrors:            []string{"resourceVersionMatch: Forbidden: resourceVersionMatch is forbidden when continue is provided"},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateListOptions(&tc.opts, tc.watchListFeatureEnabled)
			if len(tc.expectErrors) > 0 {
				if len(errs) != len(tc.expectErrors) {
					t.Errorf("expected %d errors but got %d errors", len(tc.expectErrors), len(errs))
					return
				}
				for i, expectedErr := range tc.expectErrors {
					if expectedErr != errs[i].Error() {
						t.Errorf("expected error '%s' but got '%s'", expectedErr, errs[i].Error())
					}
				}
				return
			}
			if len(errs) != 0 {
				t.Errorf("expected no errors, but got: %v", errs)
			}
		})
	}
}
