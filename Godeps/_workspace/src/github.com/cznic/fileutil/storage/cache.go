// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"container/list"
	"io"
	"math"
	"os"
	"sync"
	"sync/atomic"
)

type cachepage struct {
	b     [512]byte
	dirty bool
	lru   *list.Element
	pi    int64
	valid int // page content is b[:valid]
}

func (p *cachepage) wr(b []byte, off int) (wasDirty bool) {
	copy(p.b[off:], b)
	if n := off + len(b); n > p.valid {
		p.valid = n
	}
	wasDirty = p.dirty
	p.dirty = true
	return
}

func (c *Cache) rd(off int64, read bool) (p *cachepage, ok bool) {
	c.Rq++
	pi := off >> 9
	if p, ok = c.m[pi]; ok {
		c.lru.MoveToBack(p.lru)
		return
	}

	if !read {
		return
	}

	fp := off &^ 511
	if fp >= c.size {
		return
	}

	rq := 512
	if fp+512 > c.size {
		rq = int(c.size - fp)
	}
	p = &cachepage{pi: pi, valid: rq}
	p.lru = c.lru.PushBack(p)
	if n, err := c.f.ReadAt(p.b[:p.valid], fp); n != rq {
		panic(err)
	}

	c.Load++
	if c.advise != nil {
		c.advise(fp, 512, false)
	}
	c.m[pi], ok = p, true
	return
}

func (c *Cache) wr(off int64) (p *cachepage) {
	var ok bool
	if p, ok = c.rd(off, false); ok {
		return
	}

	pi := off >> 9
	p = &cachepage{pi: pi}
	p.lru = c.lru.PushBack(p)
	c.m[pi] = p
	return
}

// Cache provides caching support for another store Accessor.
type Cache struct {
	advise   func(int64, int, bool)
	clean    chan bool
	cleaning int32
	close    chan bool
	f        Accessor
	fi       *FileInfo
	lock     sync.Mutex
	lru      *list.List
	m        map[int64]*cachepage
	maxpages int
	size     int64
	sync     chan bool
	wlist    *list.List
	write    chan bool
	writing  int32
	Rq       int64 // Pages requested from cache
	Load     int64 // Pages loaded (cache miss)
	Purge    int64 // Pages purged
	Top      int   // "High water" pages
}

// Implementation of Accessor.
func (c *Cache) BeginUpdate() error { return nil }

// Implementation of Accessor.
func (c *Cache) EndUpdate() error { return nil }

// NewCache creates a caching Accessor from store with total of maxcache bytes.
// NewCache returns the new Cache, implementing Accessor or an error if any.
//
// The LRU mechanism is used, so the cache tries to keep often accessed pages cached.
//
func NewCache(store Accessor, maxcache int64, advise func(int64, int, bool)) (c *Cache, err error) {
	var fi os.FileInfo
	if fi, err = store.Stat(); err != nil {
		return
	}

	x := maxcache >> 9
	if x > math.MaxInt32/2 {
		x = math.MaxInt32 / 2
	}
	c = &Cache{
		advise:   advise,
		clean:    make(chan bool, 1),
		close:    make(chan bool),
		f:        store,
		lru:      list.New(), // front == oldest used, back == last recently used
		m:        make(map[int64]*cachepage),
		maxpages: int(x),
		size:     fi.Size(),
		sync:     make(chan bool),
		wlist:    list.New(),
		write:    make(chan bool, 1),
	}
	c.fi = NewFileInfo(fi, c)
	go c.writer()
	go c.cleaner(int((int64(c.maxpages) * 95) / 100)) // hysteresis
	return
}

func (c *Cache) Accessor() Accessor {
	return c.f
}

func (c *Cache) Close() (err error) {
	close(c.write)
	<-c.close
	close(c.clean)
	<-c.close
	return c.f.Close()
}

func (c *Cache) Name() (s string) {
	return c.f.Name()
}

func (c *Cache) ReadAt(b []byte, off int64) (n int, err error) {
	po := int(off) & 0x1ff
	bp := 0
	rem := len(b)
	m := 0
	for rem != 0 {
		c.lock.Lock() // X1+
		p, ok := c.rd(off, true)
		if !ok {
			c.lock.Unlock() // X1-
			return -1, io.EOF
		}

		rq := rem
		if po+rq > 512 {
			rq = 512 - po
		}
		if n := copy(b[bp:bp+rq], p.b[po:p.valid]); n != rq {
			c.lock.Unlock() // X1-
			return -1, io.EOF
		}

		m = len(c.m)
		c.lock.Unlock() // X1-
		po = 0
		bp += rq
		off += int64(rq)
		rem -= rq
		n += rq
	}
	if m > c.maxpages && atomic.CompareAndSwapInt32(&c.cleaning, 0, 1) {
		if m > c.Top {
			c.Top = m
		}
		c.clean <- true
	}
	return
}

func (c *Cache) Stat() (fi os.FileInfo, err error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.fi, nil
}

func (c *Cache) Sync() (err error) {
	c.write <- false
	<-c.sync
	return
}

func (c *Cache) Truncate(size int64) (err error) {
	c.Sync() //TODO improve (discard pages, the writer goroutine should also be aware, ...)
	c.lock.Lock()
	defer c.lock.Unlock()
	c.size = size
	return c.f.Truncate(size)
}

func (c *Cache) WriteAt(b []byte, off int64) (n int, err error) {
	po := int(off) & 0x1ff
	bp := 0
	rem := len(b)
	m := 0
	for rem != 0 {
		c.lock.Lock() // X+
		p := c.wr(off)
		rq := rem
		if po+rq > 512 {
			rq = 512 - po
		}
		if wasDirty := p.wr(b[bp:bp+rq], po); !wasDirty {
			c.wlist.PushBack(p)
		}
		m = len(c.m)
		po = 0
		bp += rq
		off += int64(rq)
		if off > c.size {
			c.size = off
		}
		c.lock.Unlock() // X-
		rem -= rq
		n += rq
	}
	if atomic.CompareAndSwapInt32(&c.writing, 0, 1) {
		c.write <- true
	}
	if m > c.maxpages && atomic.CompareAndSwapInt32(&c.cleaning, 0, 1) {
		if m > c.Top {
			c.Top = m
		}
		c.clean <- true
	}
	return
}

func (c *Cache) writer() {
	for ok := true; ok; {
		var wr bool
		var off int64
		wr, ok = <-c.write
		for {
			c.lock.Lock() // X1+
			item := c.wlist.Front()
			if item == nil {
				c.lock.Unlock() // X1-
				break
			}

			p := item.Value.(*cachepage)
			off = p.pi << 9
			if n, err := c.f.WriteAt(p.b[:p.valid], off); n != p.valid {
				c.lock.Unlock()                    // X1-
				panic("TODO Cache.writer errchan") //TODO +errchan
				panic(err)
			}

			p.dirty = false
			c.wlist.Remove(item)
			if c.advise != nil {
				c.advise(off, 512, true)
			}
			c.lock.Unlock() // X1-
		}
		switch {
		case wr:
			atomic.AddInt32(&c.writing, -1)
		case ok:
			c.sync <- true
		}
	}
	c.close <- true
}

func (c *Cache) cleaner(limit int) {
	for _ = range c.clean {
		var item *list.Element
		for {
			c.lock.Lock() // X1+
			if len(c.m) < limit {
				c.lock.Unlock() // X1-
				break
			}

			if item == nil {
				item = c.lru.Front()
			}
			if p := item.Value.(*cachepage); !p.dirty {
				delete(c.m, p.pi)
				c.lru.Remove(item)
				c.Purge++
			}
			item = item.Next()
			c.lock.Unlock() // X1-
		}
		atomic.AddInt32(&c.cleaning, -1)
	}
	c.close <- true
}
