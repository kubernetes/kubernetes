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

package coverage

import (
	"fmt"
	"strings"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// CoverageTracker collects and classifies operations against the 80 API state matrix.
type CoverageTracker struct {
	mu        sync.Mutex
	versioner storage.Versioner
	counts    map[Category]int64
}

// NewTracker creates a new thread-safe coverage tracker.
func NewTracker() *CoverageTracker {
	counts := make(map[Category]int64)
	for _, sec := range SectionCategories {
		for _, category := range sec.Categories {
			counts[category] = 0
		}
	}
	return &CoverageTracker{
		versioner: storage.APIObjectVersioner{},
		counts:    counts,
	}
}

func (t *CoverageTracker) inc(cat Category) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.counts[cat]++
}

func (t *CoverageTracker) RecordGet(key string, opts storage.GetOptions, out runtime.Object, err error) {
	// 1. RV Classification
	switch opts.ResourceVersion {
	case "":
		t.inc(CategoryGetRVQuorumLatest)
	case "0":
		t.inc(CategoryGetRVCacheAny)
	case "abc", "invalid":
		t.inc(CategoryGetRVMalformed)
	default:
		if rv, parseErr := t.versioner.ParseResourceVersion(opts.ResourceVersion); parseErr == nil {
			if rv > 1000000 {
				t.inc(CategoryGetRVFuture)
			} else {
				t.inc(CategoryGetRVPastSnapshot)
			}
		} else {
			t.inc(CategoryGetRVMalformed)
		}
	}

	// 2. Options
	if opts.IgnoreNotFound {
		t.inc(CategoryGetOptIgnoreNotFoundTrue)
	} else {
		t.inc(CategoryGetOptIgnoreNotFoundFalse)
	}

	// 3. Outcomes
	if err == nil {
		if out != nil {
			t.inc(CategoryGetKeyExisting)
			t.inc(CategoryGetOutcomeFound)
		} else {
			t.inc(CategoryGetKeyNonExisting)
			t.inc(CategoryGetOutcomeIgnoredNotFound)
		}
	} else {
		if storage.IsNotFound(err) || apierrors.IsNotFound(err) {
			t.inc(CategoryGetKeyNonExisting)
			t.inc(CategoryGetOutcomeNotFound)
		} else if storage.IsTooLargeResourceVersion(err) {
			t.inc(CategoryGetOutcomeErrTooLarge)
		} else if apierrors.IsResourceExpired(err) {
			t.inc(CategoryGetRVCompacted)
			t.inc(CategoryGetOutcomeErrTooOld)
		} else if storage.IsInvalidObj(err) || apierrors.IsBadRequest(err) {
			t.inc(CategoryGetOutcomeErrInvalid)
		}
	}
}

func (t *CoverageTracker) RecordGetList(key string, opts storage.ListOptions, out runtime.Object, err error) {
	// 1. RV Classification
	switch opts.ResourceVersion {
	case "":
		t.inc(CategoryListRVQuorumLatest)
	case "0":
		t.inc(CategoryListRVCacheAny)
	case "abc", "invalid":
		t.inc(CategoryListRVMalformed)
	default:
		if rv, parseErr := t.versioner.ParseResourceVersion(opts.ResourceVersion); parseErr == nil {
			if rv > 1000000 {
				t.inc(CategoryListRVFuture)
			} else {
				t.inc(CategoryListRVPastSnapshot)
			}
		} else {
			t.inc(CategoryListRVMalformed)
		}
	}

	// 2. Match Mode
	switch opts.ResourceVersionMatch {
	case "":
		t.inc(CategoryListMatchUnset)
	case metav1.ResourceVersionMatchExact:
		t.inc(CategoryListMatchExact)
	case metav1.ResourceVersionMatchNotOlderThan:
		t.inc(CategoryListMatchNotOlder)
	default:
		t.inc(CategoryListMatchInvalid)
	}

	// 3. Pagination
	limit := opts.Predicate.Limit
	continueToken := opts.Predicate.Continue
	if limit == 0 && continueToken == "" {
		t.inc(CategoryListPagingUnpaged)
	} else if limit > 0 && continueToken == "" {
		t.inc(CategoryListPagingInitChunk)
	} else if continueToken != "" {
		t.inc(CategoryListPagingContinued)
	}
	if continueToken != "" && opts.ResourceVersion != "" && opts.ResourceVersion != "0" {
		t.inc(CategoryListPagingConflict)
	}

	// 4. Scope
	trimmed := strings.Trim(key, "/")
	parts := strings.Split(trimmed, "/")
	if !opts.Recursive {
		t.inc(CategoryListScopeSingleKey)
	} else if len(parts) <= 1 || trimmed == "pods" {
		t.inc(CategoryListScopeRoot)
	} else {
		t.inc(CategoryListScopeSubtree)
	}

	// 5. Predicate / Filters
	if opts.Predicate.Empty() {
		t.inc(CategoryListFilterEverything)
	}
	if opts.Predicate.Label != nil && !opts.Predicate.Label.Empty() {
		t.inc(CategoryListFilterLabelMatch)
	}
	if opts.Predicate.Field != nil && !opts.Predicate.Field.Empty() {
		t.inc(CategoryListFilterFieldMatch)
	}

	// 6. Outcomes
	if err == nil {
		itemCount := 0
		if listObj, ok := out.(metav1.ListInterface); ok {
			_ = listObj
		}
		if list, listErr := meta.ExtractList(out); listErr == nil {
			itemCount = len(list)
		}
		if continueToken != "" || limit > 0 {
			t.inc(CategoryListOutcomeSuccessPaged)
		} else if itemCount > 0 {
			t.inc(CategoryListOutcomeSuccessFull)
		} else {
			t.inc(CategoryListOutcomeSuccessEmpty)
			t.inc(CategoryListFilterZeroMatch)
		}
	} else {
		if storage.IsTooLargeResourceVersion(err) {
			t.inc(CategoryListOutcomeErrTooLarge)
		} else if apierrors.IsResourceExpired(err) {
			t.inc(CategoryListRVCompacted)
			t.inc(CategoryListOutcomeErrTooOld)
		} else if storage.IsInvalidObj(err) || apierrors.IsBadRequest(err) {
			t.inc(CategoryListOutcomeErrInvalid)
		}
	}
}

func (t *CoverageTracker) RecordWatch(key string, opts storage.ListOptions, err error) {
	if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
		t.inc(CategoryWatchModeWatchListInitial)
	} else if opts.ResourceVersion == "" {
		t.inc(CategoryWatchModeStandardLive)
	} else if opts.ResourceVersion == "0" || opts.ResourceVersion == "1" {
		t.inc(CategoryWatchModeFromZeroHistory)
	} else {
		t.inc(CategoryWatchModeHistoricalRV)
	}

	trimmed := strings.Trim(key, "/")
	parts := strings.Split(trimmed, "/")
	if !opts.Recursive {
		t.inc(CategoryWatchScopeSingleKey)
	} else if len(parts) <= 1 || trimmed == "pods" {
		t.inc(CategoryWatchScopeRootPrefix)
	} else {
		t.inc(CategoryWatchScopeSubtree)
	}

	if opts.Predicate.Empty() {
		t.inc(CategoryWatchFilterEverything)
	}
	if opts.Predicate.Label != nil && !opts.Predicate.Label.Empty() {
		t.inc(CategoryWatchFilterLabelMatch)
	}
	if opts.Predicate.Field != nil && !opts.Predicate.Field.Empty() {
		t.inc(CategoryWatchFilterFieldMatch)
	}

	if err != nil {
		if apierrors.IsResourceExpired(err) {
			t.inc(CategoryWatchModeCompactedRV)
			t.inc(CategoryWatchOutcomeCompactedErr)
		}
	}
}

func (t *CoverageTracker) RecordWatchEvent(ev watch.Event) {
	switch ev.Type {
	case watch.Added:
		t.inc(CategoryWatchEventAdded)
		t.inc(CategoryWatchOutcomeEventStream)
	case watch.Modified:
		t.inc(CategoryWatchEventModified)
		t.inc(CategoryWatchOutcomeEventStream)
	case watch.Deleted:
		t.inc(CategoryWatchEventDeleted)
		t.inc(CategoryWatchOutcomeEventStream)
	case watch.Bookmark:
		isInitialEventsEnd := false
		if metaObj, ok := ev.Object.(metav1.Object); ok {
			if metaObj.GetAnnotations() != nil && metaObj.GetAnnotations()["k8s.io/initial-events-end"] == "true" {
				isInitialEventsEnd = true
			}
		}
		if isInitialEventsEnd {
			t.inc(CategoryWatchBookmarkInitialEnd)
		} else {
			t.inc(CategoryWatchBookmarkProgress)
		}
	case watch.Error:
		t.inc(CategoryWatchOutcomeCompactedErr)
	}
}

func (t *CoverageTracker) RecordCreate(key string, obj runtime.Object, out runtime.Object, ttl uint64, err error) {
	t.inc(CategoryMutCreate)
	t.inc(CategoryMutPrecondNone)
	if err == nil {
		t.inc(CategoryMutOutcomeSuccess)
	} else if storage.IsExist(err) || apierrors.IsAlreadyExists(err) {
		t.inc(CategoryMutOutcomeKeyConflict)
	} else if storage.IsInvalidObj(err) || apierrors.IsInvalid(err) {
		t.inc(CategoryMutOutcomePrecondFail)
	}
}

func (t *CoverageTracker) RecordUpdate(key string, ignoreNotFound bool, preconds *storage.Preconditions, isMutating bool, isShortCircuit bool, hasCachedObj bool, err error) {
	t.inc(CategoryMutUpdate)

	if preconds == nil {
		t.inc(CategoryMutPrecondNone)
	} else if preconds.ResourceVersion != nil {
		if *preconds.ResourceVersion == "1" {
			t.inc(CategoryMutPrecondStaleRV)
		} else {
			t.inc(CategoryMutPrecondMatchingRV)
		}
	}

	if ignoreNotFound {
		t.inc(CategoryMutUpdateUpsert)
	}
	if hasCachedObj {
		t.inc(CategoryMutUpdateCachedObj)
	}
	if isShortCircuit {
		t.inc(CategoryMutUpdateShortCircuit)
	} else if isMutating {
		t.inc(CategoryMutUpdateMutating)
	}

	if err == nil {
		t.inc(CategoryMutOutcomeSuccess)
	} else if storage.IsNotFound(err) || apierrors.IsNotFound(err) {
		t.inc(CategoryMutOutcomeNotFound)
	} else if storage.IsConflict(err) || storage.IsInvalidObj(err) || apierrors.IsConflict(err) {
		t.inc(CategoryMutOutcomePrecondFail)
	}
}

func (t *CoverageTracker) RecordDelete(key string, preconds *storage.Preconditions, validateReject bool, err error) {
	t.inc(CategoryMutDelete)

	if preconds == nil {
		t.inc(CategoryMutPrecondNone)
	} else if preconds.ResourceVersion != nil {
		t.inc(CategoryMutPrecondMatchingRV)
	}

	if validateReject {
		t.inc(CategoryMutDeleteValidationReject)
	}

	if err == nil {
		t.inc(CategoryMutOutcomeSuccess)
	} else if storage.IsNotFound(err) || apierrors.IsNotFound(err) {
		t.inc(CategoryMutDeleteNotFound)
		t.inc(CategoryMutOutcomeNotFound)
	} else if storage.IsInvalidObj(err) {
		t.inc(CategoryMutOutcomeRejectFail)
	} else if storage.IsConflict(err) {
		t.inc(CategoryMutOutcomePrecondFail)
	}
}

// FormatScorecard renders the category matrix report as formatted plain text.
// If sections are provided, only those sections are included in the report.
func (t *CoverageTracker) FormatScorecard(sections ...Section) string {
	t.mu.Lock()
	defer t.mu.Unlock()

	filter := make(map[Section]bool, len(sections))
	for _, s := range sections {
		filter[s] = true
	}

	var sb strings.Builder
	sb.WriteString("\n========================================================================\n")
	sb.WriteString("              KUBERNETES STORAGE API STATE COVERAGE SCORECARD            \n")
	sb.WriteString("========================================================================\n")
	sb.WriteString(fmt.Sprintf("%-50s | %8s | %s\n", "API State Category", "Hits", "Status"))
	sb.WriteString("---------------------------------------------------+----------+---------\n")

	totalCategories := 0
	coveredCategories := 0

	for _, sec := range SectionCategories {
		if len(filter) > 0 && !filter[sec.Section] {
			continue
		}
		sb.WriteString(fmt.Sprintf(">>> %s\n", sec.Section))
		sb.WriteString("---------------------------------------------------+----------+---------\n")
		for _, cat := range sec.Categories {
			totalCategories++
			count := t.counts[cat]
			status := "[ ] Missing"
			if count > 0 {
				coveredCategories++
				status = fmt.Sprintf("[✓] Covered (%d)", count)
			}
			sb.WriteString(fmt.Sprintf("%-50s | %8d | %s\n", cat, count, status))
		}
		sb.WriteString("---------------------------------------------------+----------+---------\n")
	}

	pct := 0.0
	if totalCategories > 0 {
		pct = (float64(coveredCategories) / float64(totalCategories)) * 100.0
	}

	sb.WriteString("========================================================================\n")
	sb.WriteString(fmt.Sprintf("TOTAL API STATES COVERED: %d / %d (%.1f%%)\n", coveredCategories, totalCategories, pct))
	sb.WriteString("========================================================================\n")

	return sb.String()
}
