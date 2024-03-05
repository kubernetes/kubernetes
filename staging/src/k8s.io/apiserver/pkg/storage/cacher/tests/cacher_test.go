/*
Copyright 2015 The Kubernetes Authors.

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

// Package tests contains cacher tests that run embedded etcd. This is to avoid dependency on "testing" in cacher package.
package tests

import (
	"context"
	"testing"

	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

func init() {
	InitTestSchema()
}

func checkStorageInvariants(ctx context.Context, t *testing.T, key string) {
	// No-op function since cacher simply passes object creation to the underlying storage.
}

func TestCreate(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreate(ctx, t, c, checkStorageInvariants)
}

func TestCreateWithTTL(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithTTL(ctx, t, c)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithKeyExist(ctx, t, c)
}

func TestGet(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGet(ctx, t, c)
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestUnconditionalDelete(ctx, t, c)
}

func TestConditionalDelete(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestConditionalDelete(ctx, t, c)
}

func TestDeleteWithSuggestion(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestion(ctx, t, c)
}

func TestDeleteWithSuggestionAndConflict(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestionAndConflict(ctx, t, c)
}

func TestDeleteWithSuggestionOfDeletedObject(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(ctx, t, c)
}

func TestValidateDeletionWithSuggestion(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestValidateDeletionWithSuggestion(ctx, t, c)
}

func TestValidateDeletionWithOnlySuggestionValid(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestValidateDeletionWithOnlySuggestionValid(ctx, t, c)
}

func TestDeleteWithConflict(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithConflict(ctx, t, c)
}

func TestPreconditionalDeleteWithSuggestion(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestPreconditionalDeleteWithSuggestion(ctx, t, c)
}

func TestPreconditionalDeleteWithSuggestionPass(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestPreconditionalDeleteWithOnlySuggestionPass(ctx, t, c)
}

func TestGetListNonRecursive(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGetListNonRecursive(ctx, t, c)
}

func checkStorageCalls(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
	// No-op function for now, since cacher passes pagination calls to underlying storage.
}

func TestListContinuation(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuation(ctx, t, c, checkStorageCalls)
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaginationRareObject(ctx, t, c, checkStorageCalls)
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuationWithFilter(ctx, t, c, checkStorageCalls)
}

func TestListInconsistentContinuation(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	// TODO(#109831): Enable use of this by setting compaction.
	storagetesting.RunTestListInconsistentContinuation(ctx, t, c, nil)
}

func TestConsistentList(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdate(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, c)
}

func TestGuaranteedUpdateChecksStoredData(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdateWithConflict(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithConflict(ctx, t, c)
}

func TestGuaranteedUpdateWithSuggestionAndConflict(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx, t, c)
}

func TestTransformationFailure(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestCount(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCount(ctx, t, c)
}

func TestWatch(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatch(ctx, t, c)
}

func TestWatchFromZero(t *testing.T) {
	ctx, c, server, terminate := TestSetupWithEtcdServer(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchFromZero(ctx, t, c, CompactStorage(c, server.V3Client))
}

func TestDeleteTriggerWatch(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteTriggerWatch(ctx, t, c)
}

func TestWatchFromNonZero(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchFromNonZero(ctx, t, c)
}

func TestDelayedWatchDelivery(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDelayedWatchDelivery(ctx, t, c)
}

func TestWatchError(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestWatchContextCancel(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestWatcherTimeout(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatcherTimeout(ctx, t, c)
}

func TestWatchDeleteEventObjectHaveLatestRV(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, c)
}

func TestWatchInitializationSignal(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchInitializationSignal(ctx, t, c)
}

func TestClusterScopedWatch(t *testing.T) {
	ctx, c, terminate := TestSetup(t, withClusterScopedKeyFunc, withSpecNodeNameIndexerFuncs)
	t.Cleanup(terminate)
	storagetesting.RunTestClusterScopedWatch(ctx, t, c)
}

func TestNamespaceScopedWatch(t *testing.T) {
	ctx, c, terminate := TestSetup(t, withSpecNodeNameIndexerFuncs)
	t.Cleanup(terminate)
	storagetesting.RunTestNamespaceScopedWatch(ctx, t, c)
}

func TestWatchDispatchBookmarkEvents(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchDispatchBookmarkEvents(ctx, t, c, true)
}

func TestWatchBookmarksWithCorrectResourceVersion(t *testing.T) {
	ctx, c, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestOptionalWatchBookmarksWithCorrectResourceVersion(ctx, t, c)
}

func TestSendInitialEventsBackwardCompatibility(t *testing.T) {
	ctx, store, terminate := TestSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunSendInitialEventsBackwardCompatibility(ctx, t, store)
}

func TestWatchSemantics(t *testing.T) {
	store, terminate := testSetupWithEtcdAndCreateWrapper(t)
	t.Cleanup(terminate)
	storagetesting.RunWatchSemantics(context.TODO(), t, store)
}

func TestWatchSemanticInitialEventsExtended(t *testing.T) {
	store, terminate := testSetupWithEtcdAndCreateWrapper(t)
	t.Cleanup(terminate)
	storagetesting.RunWatchSemanticInitialEventsExtended(context.TODO(), t, store)
}
