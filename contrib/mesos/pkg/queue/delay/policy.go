/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package delay

// Decide whether an item in a delay-queue should be rescheduled if an item with the
// same UniqueID is added/offered. Rescheduling does not effect the Item value.
type SchedulingPolicy int

const (
	PreferLatest SchedulingPolicy = iota
	PreferEarliest
)

func (dp SchedulingPolicy) EventTime(a, b Priority) (result Priority) {
	switch dp {
	case PreferEarliest:
		if a.eventTime.Before(b.eventTime) {
			result = a
		} else {
			result = b
		}
	case PreferLatest:
		fallthrough
	default:
		if a.eventTime.After(b.eventTime) {
			result = a
		} else {
			result = b
		}
	}
	return
}

// Decide whether an item in a delay-queue should have its value replaced if an item
// with the same UniqueID is added/offered. Value replacement does not effect the
// Item event time.
type ValueReplacementPolicy int

const (
	KeepExisting ValueReplacementPolicy = iota
	ReplaceExisting
)

func (rp ValueReplacementPolicy) Value(old, new interface{}) (result interface{}) {
	switch rp {
	case KeepExisting:
		result = old
	case ReplaceExisting:
		fallthrough
	default:
		result = new
	}
	return
}
