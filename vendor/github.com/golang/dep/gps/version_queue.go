// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"strings"
)

type failedVersion struct {
	v Version
	f error
}

type versionQueue struct {
	id           ProjectIdentifier
	pi           []Version
	lockv, prefv Version
	fails        []failedVersion
	b            sourceBridge
	failed       bool
	allLoaded    bool
	adverr       error
}

func newVersionQueue(id ProjectIdentifier, lockv, prefv Version, b sourceBridge) (*versionQueue, error) {
	vq := &versionQueue{
		id: id,
		b:  b,
	}

	// Lock goes in first, if present
	if lockv != nil {
		vq.lockv = lockv
		vq.pi = append(vq.pi, lockv)
	}

	// Preferred version next
	if prefv != nil {
		vq.prefv = prefv
		vq.pi = append(vq.pi, prefv)
	}

	if len(vq.pi) == 0 {
		var err error
		vq.pi, err = vq.b.listVersions(vq.id)
		if err != nil {
			// TODO(sdboyer) pushing this error this early entails that we
			// unconditionally deep scan (e.g. vendor), as well as hitting the
			// network.
			return nil, err
		}
		vq.allLoaded = true
	}

	return vq, nil
}

func (vq *versionQueue) current() Version {
	if len(vq.pi) > 0 {
		return vq.pi[0]
	}

	return nil
}

// advance moves the versionQueue forward to the next available version,
// recording the failure that eliminated the current version.
func (vq *versionQueue) advance(fail error) error {
	// Nothing in the queue means...nothing in the queue, nicely enough
	if vq.adverr != nil || len(vq.pi) == 0 { // should be a redundant check, but just in case
		return vq.adverr
	}

	// Record the fail reason and pop the queue
	vq.fails = append(vq.fails, failedVersion{
		v: vq.pi[0],
		f: fail,
	})
	vq.pi = vq.pi[1:]

	// *now*, if the queue is empty, ensure all versions have been loaded
	if len(vq.pi) == 0 {
		if vq.allLoaded {
			// This branch gets hit when the queue is first fully exhausted,
			// after a previous advance() already called ListVersions().
			return nil
		}
		vq.allLoaded = true

		var vltmp []Version
		vltmp, vq.adverr = vq.b.listVersions(vq.id)
		if vq.adverr != nil {
			return vq.adverr
		}
		// defensive copy - calling listVersions here means slice contents may
		// be modified when removing prefv/lockv.
		vq.pi = make([]Version, len(vltmp))
		copy(vq.pi, vltmp)

		// search for and remove lockv and prefv, in a pointer GC-safe manner
		//
		// could use the version comparator for binary search here to avoid
		// O(n) each time...if it matters
		var delkeys []int
		for k, pi := range vq.pi {
			if pi == vq.lockv || pi == vq.prefv {
				delkeys = append(delkeys, k)
			}
		}

		for k, dk := range delkeys {
			dk -= k
			copy(vq.pi[dk:], vq.pi[dk+1:])
			// write nil to final position for GC safety
			vq.pi[len(vq.pi)-1] = nil
			vq.pi = vq.pi[:len(vq.pi)-1]
		}

		if len(vq.pi) == 0 {
			// If listing versions added nothing (new), then return now
			return nil
		}
	}

	// We're finally sure that there's something in the queue. Remove the
	// failure marker, as the current version may have failed, but the next one
	// hasn't yet
	vq.failed = false

	// If all have been loaded and the queue is empty, we're definitely out
	// of things to try. Return empty, though, because vq semantics dictate
	// that we don't explicitly indicate the end of the queue here.
	return nil
}

// isExhausted indicates whether or not the queue has definitely been exhausted,
// in which case it will return true.
//
// It may return false negatives - suggesting that there is more in the queue
// when a subsequent call to current() will be empty. Plan accordingly.
func (vq *versionQueue) isExhausted() bool {
	if !vq.allLoaded {
		return false
	}
	return len(vq.pi) == 0
}

func (vq *versionQueue) String() string {
	var vs []string

	for _, v := range vq.pi {
		vs = append(vs, v.String())
	}
	return fmt.Sprintf("[%s]", strings.Join(vs, ", "))
}
