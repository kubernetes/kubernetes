// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package cache

// LRU cache that holds RRs and for DNSSEC an RRSIG.

// TODO(miek): try to kill the mutex or at least don't write when we read.
// TODO(miek): split elem in a rrsig and msg one so we store RRSIGs more efficient.

import (
	"container/list"
	"crypto/sha1"
	"sync"
	"time"

	"github.com/miekg/dns"
)

// Elem hold an answer and additional section that returned from the cache.
// The signature is put in answer, extra is empty there. This wastes some memory.
type elem struct {
	key        string
	expiration time.Time // time added + TTL, after this the elem is invalid
	msg        *dns.Msg
}

// Cache is a LRU cache that holds on the a number of RRs or DNS messgas.
type Cache struct {
	sync.Mutex
	l        *list.List
	m        map[string]*list.Element
	capacity uint // max number of RRs
	size     uint // current size
	ttl      time.Duration
}

// TODO(miek): add setCapacity so it can be set runtime.
// TODO(miek): makes this lockfree(er).

// New returns a new cache with the capacity and the ttl specified.
func New(capacity, ttl int) *Cache {
	c := new(Cache)
	c.l = list.New()
	c.m = make(map[string]*list.Element)
	c.capacity = uint(capacity)
	c.ttl = time.Duration(ttl) * time.Second
	return c
}

// Size returns the size of the cache as an int.
func (c *Cache) Size() int { return int(c.size) }

// Capacity returns the capacity of the cache as an int.
func (c *Cache) Capacity() int { return int(c.capacity) }

// Remove removes the element under key s from the cache.
func (c *Cache) Remove(s string) {
	c.Lock()
	defer c.Unlock()
	e := c.m[s]
	if e == nil {
		return
	}
	c.size -= 1
	c.l.Remove(e)
	delete(c.m, s)
	c.shrink()
}

// Shrink shrinks the cache back to its capacity.
func (c *Cache) shrink() {
	for c.size > c.capacity {
		e := c.l.Back()
		if e == nil { // nothing left
			break
		}
		v := e.Value.(*elem)
		c.l.Remove(e)
		delete(c.m, v.key)
		c.size -= uint(len(v.msg.Answer) + len(v.msg.Ns) + len(v.msg.Extra))
	}
}

// InsertMessage inserts a message in the Cache. We will cache it for ttl seconds, which
// should be a small (60...300) integer.
func (c *Cache) InsertMessage(s string, msg *dns.Msg) {
	if c.capacity == 0 {
		return
	}
	c.Lock()
	defer c.Unlock()
	if _, ok := c.m[s]; !ok {
		e := c.l.PushFront(&elem{s, time.Now().UTC().Add(c.ttl), msg})
		c.m[s] = e
	}
	c.size += uint(len(msg.Answer) + len(msg.Ns) + len(msg.Extra))
	c.shrink()
}

// InsertSignature inserts a signature, the expiration time is used as the cache ttl.
func (c *Cache) InsertSignature(s string, sig *dns.RRSIG) {
	if c.capacity == 0 {
		return
	}
	c.Lock()
	defer c.Unlock()
	if _, ok := c.m[s]; !ok {
		m := ((int64(sig.Expiration) - time.Now().Unix()) / (1 << 31)) - 1
		if m < 0 {
			m = 0
		}
		t := time.Unix(int64(sig.Expiration)-(m*(1<<31)), 0).UTC()
		e := c.l.PushFront(&elem{s, t, &dns.Msg{Answer: []dns.RR{sig}}})
		c.m[s] = e
	}
	c.size += 1
	c.shrink()
}

// Search returns .... and a boolean indicating if we found something
// in the cache.
func (c *Cache) Search(s string) (*dns.Msg, time.Time, bool) {
	if c.capacity == 0 {
		return nil, time.Time{}, false
	}
	c.Lock()
	defer c.Unlock()
	if e, ok := c.m[s]; ok {
		c.l.MoveToFront(e)
		e := e.Value.(*elem)
		e1 := e.msg.Copy()
		return e1, e.expiration, true
	}
	return nil, time.Time{}, false
}

// QuestionKey creates a hash key from a question section. It creates a different key
// for requests with DNSSEC.
func QuestionKey(q dns.Question, dnssec bool) string {
	h := sha1.New()
	i := append([]byte(q.Name), packUint16(q.Qtype)...)
	if dnssec {
		i = append(i, byte(255))
	}
	return string(h.Sum(i))
}

// Key uses the name, type and rdata, which is serialized and then hashed as the key for the lookup.
func Key(rrs []dns.RR) string {
	h := sha1.New()
	i := []byte(rrs[0].Header().Name)
	i = append(i, packUint16(rrs[0].Header().Rrtype)...)
	for _, r := range rrs {
		switch t := r.(type) { // we only do a few type, serialize these manually
		case *dns.SOA:
			// We only fiddle with the serial so store that.
			i = append(i, packUint32(t.Serial)...)
		case *dns.SRV:
			i = append(i, packUint16(t.Priority)...)
			i = append(i, packUint16(t.Weight)...)
			i = append(i, packUint16(t.Weight)...)
			i = append(i, []byte(t.Target)...)
		case *dns.A:
			i = append(i, []byte(t.A)...)
		case *dns.AAAA:
			i = append(i, []byte(t.AAAA)...)
		case *dns.NSEC3:
			i = append(i, []byte(t.NextDomain)...)
			// Bitmap does not differentiate in SkyDNS.
		case *dns.DNSKEY:
		case *dns.NS:
		case *dns.TXT:
		}
	}
	return string(h.Sum(i))
}

func packUint16(i uint16) []byte { return []byte{byte(i >> 8), byte(i)} }
func packUint32(i uint32) []byte { return []byte{byte(i >> 24), byte(i >> 16), byte(i >> 8), byte(i)} }
