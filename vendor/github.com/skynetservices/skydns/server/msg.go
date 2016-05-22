// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import "github.com/miekg/dns"

// Fit will make m fit the size. If a message is larger than size then entire
// additional section is dropped. If it is still to large and the transport
// is udp we return a truncated message.
// If the transport is tcp we are going to drop RR from the answer section
// until it fits. When this is case the returned bool is true.
func Fit(m *dns.Msg, size int, tcp bool) (*dns.Msg, bool) {
	if m.Len() > size {
		// Check for OPT Records at the end and keep those. TODO(miek)
		m.Extra = nil
	}
	if m.Len() < size {
		return m, false
	}

	// With TCP setting TC does not mean anything.
	if !tcp {
		m.Truncated = true
		// fall through here, so we at least return a message that can
		// fit the udp buffer.
	}

	// Additional section is gone, binary search until we have length that fits.
	min, max := 0, len(m.Answer)
	original := make([]dns.RR, len(m.Answer))
	copy(original, m.Answer)
	for {
		if min == max {
			break
		}

		mid := (min + max) / 2
		m.Answer = original[:mid]

		if m.Len() < size {
			min++
			continue
		}
		max = mid

	}
	if max > 1 {
		max--
	}
	m.Answer = m.Answer[:max]
	return m, true
}
