package encryption

import (
	"encoding/base64"

	"k8s.io/utils/lru"
)

type cache struct {
	lruCache *lru.Cache
}

func newCache(cacheSize int) *cache {
	return &cache{
		lruCache: lru.New(cacheSize),
	}
}

func (c *cache) Add(encKey []byte, aesgcm *AESGCM) {
	id := base64.StdEncoding.EncodeToString(encKey)

	c.lruCache.Add(id, aesgcm)
}

func (c *cache) Get(encKey []byte) (*AESGCM, bool) {
	id := base64.StdEncoding.EncodeToString(encKey)

	keyMaybe, ok := c.lruCache.Get(id)
	if !ok {
		return nil, false
	}

	key, ok := keyMaybe.(*AESGCM)
	if !ok {
		return nil, false
	}

	return key, true
}
