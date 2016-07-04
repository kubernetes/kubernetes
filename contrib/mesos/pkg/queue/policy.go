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

package queue

// Decide whether a pre-existing deadline for an item in a delay-queue should be
// updated if an attempt is made to offer/add a new deadline for said item. Whether
// the deadline changes or not has zero impact on the data blob associated with the
// entry in the queue.
type DeadlinePolicy int

const (
	PreferLatest DeadlinePolicy = iota
	PreferEarliest
)

// Decide whether a pre-existing data blob in a delay-queue should be replaced if an
// an attempt is made to add/offer a new data blob in its place. Whether the data is
// replaced has no bearing on the deadline (priority) of the item in the queue.
type ReplacementPolicy int

const (
	KeepExisting ReplacementPolicy = iota
	ReplaceExisting
)

func (rp ReplacementPolicy) replacementValue(original, replacement interface{}) (result interface{}) {
	switch rp {
	case KeepExisting:
		result = original
	case ReplaceExisting:
		fallthrough
	default:
		result = replacement
	}
	return
}

func (dp DeadlinePolicy) nextDeadline(a, b Priority) (result Priority) {
	switch dp {
	case PreferEarliest:
		if a.ts.Before(b.ts) {
			result = a
		} else {
			result = b
		}
	case PreferLatest:
		fallthrough
	default:
		if a.ts.After(b.ts) {
			result = a
		} else {
			result = b
		}
	}
	return
}
