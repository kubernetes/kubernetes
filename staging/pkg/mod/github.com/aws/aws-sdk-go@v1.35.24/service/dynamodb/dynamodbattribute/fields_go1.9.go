// +build go1.9

package dynamodbattribute

import "sync"

var fieldCache fieldCacher

type fieldCacher struct {
	cache sync.Map
}

func (c *fieldCacher) Load(t interface{}) (*cachedFields, bool) {
	if v, ok := c.cache.Load(t); ok {
		return v.(*cachedFields), true
	}
	return nil, false
}

func (c *fieldCacher) LoadOrStore(t interface{}, fs *cachedFields) (*cachedFields, bool) {
	v, ok := c.cache.LoadOrStore(t, fs)
	return v.(*cachedFields), ok
}
