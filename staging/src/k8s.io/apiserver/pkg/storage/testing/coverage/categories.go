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

// Category defines an individual API state category tracked in the coverage matrix.
type Category string

const (
	// =========================================================================
	// 1. GET CATEGORIES (16 categories)
	// =========================================================================
	CategoryGetRVQuorumLatest Category = "Get RV: Quorum / Latest (\"\")"
	CategoryGetRVCacheAny     Category = "Get RV: Cache-Any (\"0\")"
	CategoryGetRVPastSnapshot Category = "Get RV: Past Committed Snapshot"
	CategoryGetRVFuture       Category = "Get RV: Future / Too Large"
	CategoryGetRVCompacted    Category = "Get RV: Compacted / Too Old"
	CategoryGetRVMalformed    Category = "Get RV: Malformed (\"abc\")"

	CategoryGetKeyExisting            Category = "Get Key: Existing Object"
	CategoryGetKeyNonExisting         Category = "Get Key: Non-Existing Object"
	CategoryGetOptIgnoreNotFoundTrue  Category = "Get Option: IgnoreNotFound (true)"
	CategoryGetOptIgnoreNotFoundFalse Category = "Get Option: IgnoreNotFound (false)"

	CategoryGetOutcomeFound           Category = "Get Outcome: Found Object"
	CategoryGetOutcomeNotFound        Category = "Get Outcome: NotFound (404)"
	CategoryGetOutcomeIgnoredNotFound Category = "Get Outcome: Ignored NotFound (nil, err=nil)"
	CategoryGetOutcomeErrTooLarge     Category = "Get Outcome: Err TooLarge"
	CategoryGetOutcomeErrTooOld       Category = "Get Outcome: Err TooOld / Expired"
	CategoryGetOutcomeErrInvalid      Category = "Get Outcome: Err Invalid / Parse"

	// =========================================================================
	// 2. LIST CATEGORIES (29 categories)
	// =========================================================================
	CategoryListRVQuorumLatest Category = "List RV: Quorum / Latest (\"\")"
	CategoryListRVCacheAny     Category = "List RV: Cache-Any (\"0\")"
	CategoryListRVPastSnapshot Category = "List RV: Past Committed Snapshot"
	CategoryListRVFuture       Category = "List RV: Future / Too Large"
	CategoryListRVCompacted    Category = "List RV: Compacted / Too Old"
	CategoryListRVMalformed    Category = "List RV: Malformed (\"abc\")"

	CategoryListMatchUnset    Category = "List Match: Unset"
	CategoryListMatchExact    Category = "List Match: Exact"
	CategoryListMatchNotOlder Category = "List Match: NotOlderThan"
	CategoryListMatchInvalid  Category = "List Match: Invalid"

	CategoryListPagingUnpaged   Category = "List Pagination: Unpaged"
	CategoryListPagingInitChunk Category = "List Pagination: Initial Chunk (Limit > 0)"
	CategoryListPagingContinued Category = "List Pagination: Continue Token Drained"
	CategoryListPagingConflict  Category = "List Pagination: RV + Continue Conflict"

	CategoryListScopeRoot      Category = "List Scope: Root Prefix (/pods/)"
	CategoryListScopeSubtree   Category = "List Scope: Subtree Namespace (/pods/nsX/)"
	CategoryListScopeSingleKey Category = "List Scope: Non-Recursive Key"

	CategoryListFilterEverything Category = "List Filter: Unfiltered (Everything)"
	CategoryListFilterLabelMatch Category = "List Filter: Label Selector"
	CategoryListFilterFieldMatch Category = "List Filter: Field Selector"
	CategoryListFilterZeroMatch  Category = "List Filter: Zero Match Filter"

	CategoryListOutcomeSuccessFull  Category = "List Outcome: Success Full"
	CategoryListOutcomeSuccessPaged Category = "List Outcome: Success Paged"
	CategoryListOutcomeSuccessEmpty Category = "List Outcome: Success Empty"
	CategoryListOutcomeErrTooLarge  Category = "List Outcome: Err TooLarge"
	CategoryListOutcomeErrTooOld    Category = "List Outcome: Err TooOld / Expired"
	CategoryListOutcomeErrInvalid   Category = "List Outcome: Err Invalid / Parse"

	// =========================================================================
	// 3. WATCH CATEGORIES (18 categories)
	// =========================================================================
	CategoryWatchModeStandardLive     Category = "Watch Mode: Standard Live Stream (RV != \"\")"
	CategoryWatchModeFromZeroHistory  Category = "Watch Mode: History Stream (RV = \"0\" / \"1\")"
	CategoryWatchModeWatchListInitial Category = "Watch Mode: WatchList Snapshot (SendInitialEvents)"
	CategoryWatchModeHistoricalRV     Category = "Watch Mode: Historical Stream (RV = \"R\")"
	CategoryWatchModeCompactedRV      Category = "Watch Mode: Compacted Stream"

	CategoryWatchScopeRootPrefix Category = "Watch Scope: Root Prefix (/pods/)"
	CategoryWatchScopeSubtree    Category = "Watch Scope: Subtree Namespace (/pods/nsX/)"
	CategoryWatchScopeSingleKey  Category = "Watch Scope: Single Key"

	CategoryWatchFilterEverything Category = "Watch Filter: Unfiltered (Everything)"
	CategoryWatchFilterLabelMatch Category = "Watch Filter: Label Selector"
	CategoryWatchFilterFieldMatch Category = "Watch Filter: Field Selector"

	CategoryWatchEventAdded    Category = "Watch Event: Added (Create / Filter Enter)"
	CategoryWatchEventModified Category = "Watch Event: Modified (Update in Filter)"
	CategoryWatchEventDeleted  Category = "Watch Event: Deleted (Delete / Filter Exit)"

	CategoryWatchBookmarkInitialEnd  Category = "Watch Bookmark: initial-events-end"
	CategoryWatchBookmarkProgress    Category = "Watch Bookmark: Progress / Quiescent"
	CategoryWatchOutcomeEventStream  Category = "Watch Outcome: Continuous Event Delivery"
	CategoryWatchOutcomeCompactedErr Category = "Watch Outcome: Err TooOld / 410 Gone"

	// =========================================================================
	// 4. MUTATION CATEGORIES (17 categories)
	// =========================================================================
	CategoryMutCreate Category = "Mutation Op: Create"
	CategoryMutUpdate Category = "Mutation Op: Update"
	CategoryMutDelete Category = "Mutation Op: Delete"

	CategoryMutPrecondNone       Category = "Precondition: None (Unconditional)"
	CategoryMutPrecondMatchingRV Category = "Precondition: Matching ResourceVersion"
	CategoryMutPrecondStaleRV    Category = "Precondition: Stale / Mismatched RV"

	CategoryMutUpdateMutating         Category = "Update Path: Mutating State Change"
	CategoryMutUpdateShortCircuit     Category = "Update Path: Short-Circuit / No-Op"
	CategoryMutUpdateUpsert           Category = "Update Path: Upsert (IgnoreNotFound)"
	CategoryMutUpdateCachedObj        Category = "Update Path: CachedObject Suggestion"
	CategoryMutDeleteValidationReject Category = "Delete Path: Admission Validation Reject"
	CategoryMutDeleteNotFound         Category = "Delete Path: Key Not Found"

	CategoryMutOutcomeSuccess     Category = "Mutation Outcome: Success"
	CategoryMutOutcomeKeyConflict Category = "Mutation Outcome: Key Already Exists (409)"
	CategoryMutOutcomeNotFound    Category = "Mutation Outcome: Key Not Found (404)"
	CategoryMutOutcomePrecondFail Category = "Mutation Outcome: Precondition Failed"
	CategoryMutOutcomeRejectFail  Category = "Mutation Outcome: Validation Rejected"
)

// Section represents a group of related API state categories (Get, List, Watch, Mutation).
type Section string

const (
	SectionGet                Section = "Get"
	SectionList               Section = "List"
	SectionWatch              Section = "Watch"
	SectionCreateUpdateDelete Section = "Create / Update / Delete"
)

// SectionGroup groups categories under a section for structured reporting.
type SectionGroup struct {
	Section    Section
	Categories []Category
}

// SectionCategories groups categories by method family for structured reporting.
var SectionCategories = []SectionGroup{
	{
		Section: SectionGet,
		Categories: []Category{
			CategoryGetRVQuorumLatest,
			CategoryGetRVCacheAny,
			CategoryGetRVPastSnapshot,
			CategoryGetRVFuture,
			CategoryGetRVCompacted,
			CategoryGetRVMalformed,
			CategoryGetKeyExisting,
			CategoryGetKeyNonExisting,
			CategoryGetOptIgnoreNotFoundTrue,
			CategoryGetOptIgnoreNotFoundFalse,
			CategoryGetOutcomeFound,
			CategoryGetOutcomeNotFound,
			CategoryGetOutcomeIgnoredNotFound,
			CategoryGetOutcomeErrTooLarge,
			CategoryGetOutcomeErrTooOld,
			CategoryGetOutcomeErrInvalid,
		},
	},
	{
		Section: SectionList,
		Categories: []Category{
			CategoryListRVQuorumLatest,
			CategoryListRVCacheAny,
			CategoryListRVPastSnapshot,
			CategoryListRVFuture,
			CategoryListRVCompacted,
			CategoryListRVMalformed,
			CategoryListMatchUnset,
			CategoryListMatchExact,
			CategoryListMatchNotOlder,
			CategoryListMatchInvalid,
			CategoryListPagingUnpaged,
			CategoryListPagingInitChunk,
			CategoryListPagingContinued,
			CategoryListPagingConflict,
			CategoryListScopeRoot,
			CategoryListScopeSubtree,
			CategoryListScopeSingleKey,
			CategoryListFilterEverything,
			CategoryListFilterLabelMatch,
			CategoryListFilterFieldMatch,
			CategoryListFilterZeroMatch,
			CategoryListOutcomeSuccessFull,
			CategoryListOutcomeSuccessPaged,
			CategoryListOutcomeSuccessEmpty,
			CategoryListOutcomeErrTooLarge,
			CategoryListOutcomeErrTooOld,
			CategoryListOutcomeErrInvalid,
		},
	},
	{
		Section: SectionWatch,
		Categories: []Category{
			CategoryWatchModeStandardLive,
			CategoryWatchModeFromZeroHistory,
			CategoryWatchModeWatchListInitial,
			CategoryWatchModeHistoricalRV,
			CategoryWatchModeCompactedRV,
			CategoryWatchScopeRootPrefix,
			CategoryWatchScopeSubtree,
			CategoryWatchScopeSingleKey,
			CategoryWatchFilterEverything,
			CategoryWatchFilterLabelMatch,
			CategoryWatchFilterFieldMatch,
			CategoryWatchEventAdded,
			CategoryWatchEventModified,
			CategoryWatchEventDeleted,
			CategoryWatchBookmarkInitialEnd,
			CategoryWatchBookmarkProgress,
			CategoryWatchOutcomeEventStream,
			CategoryWatchOutcomeCompactedErr,
		},
	},
	{
		Section: SectionCreateUpdateDelete,
		Categories: []Category{
			CategoryMutCreate,
			CategoryMutUpdate,
			CategoryMutDelete,
			CategoryMutPrecondNone,
			CategoryMutPrecondMatchingRV,
			CategoryMutPrecondStaleRV,
			CategoryMutUpdateMutating,
			CategoryMutUpdateShortCircuit,
			CategoryMutUpdateUpsert,
			CategoryMutUpdateCachedObj,
			CategoryMutDeleteValidationReject,
			CategoryMutDeleteNotFound,
			CategoryMutOutcomeSuccess,
			CategoryMutOutcomeKeyConflict,
			CategoryMutOutcomeNotFound,
			CategoryMutOutcomePrecondFail,
			CategoryMutOutcomeRejectFail,
		},
	},
}
