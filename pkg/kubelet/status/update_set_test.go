/*
Copyright 2017 The Kubernetes Authors.

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

package status

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

func newTestUpdateSet() *updateSet {
	u := NewUpdateSet()
	return u.(*updateSet)
}

// updateList returns a list of updates in the same way that Updates would provide
// them in the channel.  This is useful for testing because it is deterministic.
func (u *updateSet) updateList() (updates []Update) {
	for {
		select {
		case uid := <-u.channel:
			u.lock.Lock()
			delete(u.set, uid)
			status, ok := u.cache[uid]
			u.lock.Unlock()
			if ok {
				updates = append(updates, Update{UID: uid, Status: status})
			}
		default:
			return
		}
	}
}

func expectEqualUpdates(t *testing.T, actualUpdates []Update, expectedUpdates []Update) {
	err := fmt.Errorf("Actual Updates: %+v is different from Expected Updates: %+v", actualUpdates, expectedUpdates)
	if len(actualUpdates) != len(expectedUpdates) {
		t.Error(err)
	}
	for i, expected := range expectedUpdates {
		if expected.UID != actualUpdates[i].UID || !isStatusEqual(&expected.Status, &actualUpdates[i].Status) {
			t.Error(err)
		}
	}
}

func (u *updateSet) expectGetStatus(t *testing.T, uid types.UID, expectedFound bool, expectedStatus v1.PodStatus) {
	actualStatus, found := u.Get(uid)
	if !found && expectedFound {
		t.Errorf("Expected to get a status for uid: %v, but did not", uid)
	} else if found && !expectedFound {
		t.Errorf("Did not expect to get a status for uid: %v, but got status: %+v", uid, actualStatus)
	} else if found && expectedFound && !isStatusEqual(&actualStatus, &expectedStatus) {
		t.Errorf("Actual Status: %+v was not equalto Expected Status: %+v", actualStatus, expectedStatus)
	}
}

func TestNoUpdates(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testUpdateSet.expectGetStatus(t, types.UID("uidThatDoesntExist"), false, v1.PodStatus{})
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{})
}

func TestSingleUpdate(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID := types.UID("abc")
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID,
			Status: testStatus,
		},
	})
}

func TestMultipleUpdatesToSameUID(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID := types.UID("abc")
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	// Change the status before we draw from the queue
	testStatus = getRandomPodStatus()
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID: testUID,
			// expect to see only the most recent update
			Status: testStatus,
		},
	})
}

func TestSingleUpdateToMultipleUID(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID1 := types.UID("abc")
	testUID2 := types.UID("123")
	testUpdateSet.Set(testUID1, testStatus)
	testUpdateSet.Set(testUID2, testStatus)
	testUpdateSet.expectGetStatus(t, testUID1, true, testStatus)
	testUpdateSet.expectGetStatus(t, testUID2, true, testStatus)
	// Expect to see the first update in the channel before the second update.
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID1,
			Status: testStatus,
		},
		{
			UID:    testUID2,
			Status: testStatus,
		},
	})
}

func TestMultipleUpdateToMultipleUID(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus1 := getRandomPodStatus()
	testStatus2 := getRandomPodStatus()
	testUID1 := types.UID("abc")
	testUID2 := types.UID("123")
	testUpdateSet.Set(testUID1, testStatus1)
	testUpdateSet.Set(testUID2, testStatus2)
	testUpdateSet.expectGetStatus(t, testUID1, true, testStatus1)
	testUpdateSet.expectGetStatus(t, testUID2, true, testStatus2)
	testStatus1 = getRandomPodStatus()
	testStatus2 = getRandomPodStatus()
	// Reverse the ordering, but the updates should come out of the channel in the original ordering
	testUpdateSet.Set(testUID2, testStatus2)
	testUpdateSet.Set(testUID1, testStatus1)
	testUpdateSet.expectGetStatus(t, testUID1, true, testStatus1)
	testUpdateSet.expectGetStatus(t, testUID2, true, testStatus2)
	// Expect to see the first update in the channel before the second update.
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID1,
			Status: testStatus1,
		},
		{
			UID:    testUID2,
			Status: testStatus2,
		},
	})

}

func TestDeletedUIDUpdate(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID := types.UID("abc")
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	testUpdateSet.Delete(testUID)
	testUpdateSet.expectGetStatus(t, testUID, false, v1.PodStatus{})
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{})
}

func TestRetryUpdate(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID := types.UID("abc")
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID,
			Status: testStatus,
		},
	})
	// We should have consumed the update in the previous updateList
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{})
	testUpdateSet.Retry(testUID, NoDelay)
	// After the retry, we expect to see another update!
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID,
			Status: testStatus,
		},
	})
}

func TestRetryDeletedUpdate(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID := types.UID("abc")
	testUpdateSet.Set(testUID, testStatus)
	testUpdateSet.expectGetStatus(t, testUID, true, testStatus)
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{
		{
			UID:    testUID,
			Status: testStatus,
		},
	})
	// We should have consumed the update in the previous updateList
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{})
	testUpdateSet.Retry(testUID, NoDelay)
	testUpdateSet.Delete(testUID)
	// Even after the retry, we do not expect to see an update because of the delete
	expectEqualUpdates(t, testUpdateSet.updateList(), []Update{})
}

func TestGarbageCollect(t *testing.T) {
	testUpdateSet := newTestUpdateSet()
	testStatus := getRandomPodStatus()
	testUID1 := types.UID("abc")
	testUID2 := types.UID("123")
	testUID3 := types.UID("abc123")
	testUpdateSet.Set(testUID1, testStatus)
	testUpdateSet.Set(testUID2, testStatus)
	testUpdateSet.Set(testUID3, testStatus)
	remaining := make(map[types.UID]struct{})
	remaining[testUID2] = struct{}{}
	testUpdateSet.GarbageCollect(remaining)
	testUpdateSet.expectGetStatus(t, testUID1, false, v1.PodStatus{})
	testUpdateSet.expectGetStatus(t, testUID2, true, testStatus)
	testUpdateSet.expectGetStatus(t, testUID3, false, v1.PodStatus{})
}
