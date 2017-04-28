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

package badconfig

import (
	"encoding/json"
	"fmt"
	"time"
)

// Tracker tracks "bad" configurations in a storage layer
type Tracker interface {
	// Initialize sets up the storage layer
	Initialize() error
	// MarkBad marks `uid` as a bad config and records `reason` as the reason for marking it bad
	MarkBad(uid, reason string) error
	// Entry returns the Entry for `uid` if it exists in the tracker, otherise nil
	Entry(uid string) (*Entry, error)
}

// Entry describes when a configuration was marked bad and why
type Entry struct {
	Time   string `json:"time"`
	Reason string `json:"reason"`
}

// markBad makes an entry in `m` for the config with `uid` and reason `reason`
func markBad(m map[string]Entry, uid, reason string) {
	now := time.Now()
	entry := Entry{
		Time:   now.Format(time.RFC3339), // use RFC3339 time format
		Reason: reason,
	}
	m[uid] = entry
}

// getEntry returns the Entry for `uid` in `m`, or nil if no such entry exists
func getEntry(m map[string]Entry, uid string) *Entry {
	entry, ok := m[uid]
	if ok {
		return &entry
	}
	return nil
}

// encode retuns a []byte representation of `m`, for saving `m` to a storage layer
func encode(m map[string]Entry) ([]byte, error) {
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// decode transforms a []byte into a `map[string]Entry`, or returns an error if it can't produce said map
// if `data` is empty, returns an empty map
func decode(data []byte) (map[string]Entry, error) {
	// create the map
	m := map[string]Entry{}
	// if the data is empty, just return the empty map
	if len(data) == 0 {
		return m, nil
	}
	// otherwise unmarshal the json
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("failed to unmarshal, error: %v", err)
	}
	return m, nil
}
