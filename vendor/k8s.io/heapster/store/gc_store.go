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

package store

import (
	"time"

	"github.com/golang/glog"
)

type gcStore struct {
	bufferDuration time.Duration
	store          TimeStore
}

// The GC store expects data to be inserted in reverse chronological order.
func (gcs *gcStore) Put(tp TimePoint) error {
	if err := gcs.store.Put(tp); err != nil {
		return err
	}
	gcs.reapOldData(tp.Timestamp)
	return nil
}

func (gcs *gcStore) Get(start, end time.Time) []TimePoint {
	return gcs.store.Get(start, end)
}

func (gcs *gcStore) Delete(start, end time.Time) error {
	return gcs.store.Delete(start, end)
}

func (gcs *gcStore) reapOldData(timestamp time.Time) {
	end := timestamp.Add(-gcs.bufferDuration)
	if end.Before(time.Time{}) {
		return
	}
	if err := gcs.store.Delete(time.Time{}, end); err != nil {
		glog.Errorf("failed to delete old data - %v", err)
	}
}

func NewGCStore(store TimeStore, bufferDuration time.Duration) TimeStore {
	gcStore := &gcStore{
		bufferDuration: bufferDuration,
		store:          store,
	}
	return gcStore
}
