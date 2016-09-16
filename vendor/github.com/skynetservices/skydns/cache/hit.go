// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package cache

import (
	"time"

	"github.com/miekg/dns"
)

// Hit returns a dns message from the cache. If the message's TTL is expired nil
// is returned and the message is removed from the cache.
func (c *Cache) Hit(question dns.Question, dnssec, tcp bool, msgid uint16) *dns.Msg {
	key := Key(question, dnssec, tcp)
	m1, exp, hit := c.Search(key)
	if hit {
		// Cache hit! \o/
		if time.Since(exp) < 0 {
			m1.Id = msgid
			m1.Compress = true
			// Even if something ended up with the TC bit *in* the cache, set it to off
			m1.Truncated = false
			return m1
		}
		// Expired! /o\
		c.Remove(key)
	}
	return nil
}
