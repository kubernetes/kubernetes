package fixchain

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type lockedCache struct {
	m map[string][]byte
	sync.RWMutex
}

func (c *lockedCache) get(str string) ([]byte, bool) {
	c.RLock()
	defer c.RUnlock()
	b, ok := c.m[str]
	return b, ok
}

func (c *lockedCache) set(str string, b []byte) {
	c.Lock()
	defer c.Unlock()
	c.m[str] = b
}

func newLockedCache() *lockedCache {
	return &lockedCache{m: make(map[string][]byte)}
}

type urlCache struct {
	client *http.Client
	cache  *lockedCache

	hit       uint32
	miss      uint32
	errors    uint32
	badStatus uint32
	readFail  uint32
}

func (u *urlCache) getURL(url string) ([]byte, error) {
	r, ok := u.cache.get(url)
	if ok {
		atomic.AddUint32(&u.hit, 1)
		return r, nil
	}
	c, err := u.client.Get(url)
	if err != nil {
		atomic.AddUint32(&u.errors, 1)
		return nil, err
	}
	defer c.Body.Close()
	// TODO(katjoyce): Add caching of permanent errors.
	if c.StatusCode != 200 {
		atomic.AddUint32(&u.badStatus, 1)
		return nil, fmt.Errorf("can't deal with status %d", c.StatusCode)
	}
	r, err = ioutil.ReadAll(c.Body)
	if err != nil {
		atomic.AddUint32(&u.readFail, 1)
		return nil, err
	}
	atomic.AddUint32(&u.miss, 1)
	u.cache.set(url, r)
	return r, nil
}

func newURLCache(c *http.Client, logStats bool) *urlCache {
	u := &urlCache{cache: newLockedCache(), client: c}

	if logStats {
		t := time.NewTicker(time.Second)
		go func() {
			for _ = range t.C {
				log.Printf("cache: %d hits, %d misses, %d errors, "+
					"%d bad status, %d read fail, %d cached", u.hit,
					u.miss, u.errors, u.badStatus, u.readFail,
					len(u.cache.m))
			}
		}()
	}

	return u
}
