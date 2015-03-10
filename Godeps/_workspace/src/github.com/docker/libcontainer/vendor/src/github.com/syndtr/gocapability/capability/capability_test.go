// Copyright (c) 2013, Suryandaru Triandana <syndtr@gmail.com>
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package capability

import "testing"

func TestState(t *testing.T) {
	testEmpty := func(name string, c Capabilities, whats CapType) {
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			if (i&whats) != 0 && !c.Empty(i) {
				t.Errorf(name+": capabilities set %q wasn't empty", i)
			}
		}
	}
	testFull := func(name string, c Capabilities, whats CapType) {
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			if (i&whats) != 0 && !c.Full(i) {
				t.Errorf(name+": capabilities set %q wasn't full", i)
			}
		}
	}
	testPartial := func(name string, c Capabilities, whats CapType) {
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			if (i&whats) != 0 && (c.Empty(i) || c.Full(i)) {
				t.Errorf(name+": capabilities set %q wasn't partial", i)
			}
		}
	}
	testGet := func(name string, c Capabilities, whats CapType, max Cap) {
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			if (i & whats) == 0 {
				continue
			}
			for j := Cap(0); j <= max; j++ {
				if !c.Get(i, j) {
					t.Errorf(name+": capability %q wasn't found on %q", j, i)
				}
			}
		}
	}

	capf := new(capsFile)
	capf.data.version = 2
	for _, tc := range []struct {
		name string
		c    Capabilities
		sets CapType
		max  Cap
	}{
		{"v1", new(capsV1), EFFECTIVE | PERMITTED, CAP_AUDIT_CONTROL},
		{"v3", new(capsV3), EFFECTIVE | PERMITTED | BOUNDING, CAP_LAST_CAP},
		{"file_v1", new(capsFile), EFFECTIVE | PERMITTED, CAP_AUDIT_CONTROL},
		{"file_v2", capf, EFFECTIVE | PERMITTED, CAP_LAST_CAP},
	} {
		testEmpty(tc.name, tc.c, tc.sets)
		tc.c.Fill(CAPS | BOUNDS)
		testFull(tc.name, tc.c, tc.sets)
		testGet(tc.name, tc.c, tc.sets, tc.max)
		tc.c.Clear(CAPS | BOUNDS)
		testEmpty(tc.name, tc.c, tc.sets)
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			for j := Cap(0); j <= CAP_LAST_CAP; j++ {
				tc.c.Set(i, j)
			}
		}
		testFull(tc.name, tc.c, tc.sets)
		testGet(tc.name, tc.c, tc.sets, tc.max)
		for i := CapType(1); i <= BOUNDING; i <<= 1 {
			for j := Cap(0); j <= CAP_LAST_CAP; j++ {
				tc.c.Unset(i, j)
			}
		}
		testEmpty(tc.name, tc.c, tc.sets)
		tc.c.Set(PERMITTED, CAP_CHOWN)
		testPartial(tc.name, tc.c, PERMITTED)
		tc.c.Clear(CAPS | BOUNDS)
		testEmpty(tc.name, tc.c, tc.sets)
	}
}
