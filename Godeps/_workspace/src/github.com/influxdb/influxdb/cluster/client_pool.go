package cluster

import (
	"net"
	"sync"

	"gopkg.in/fatih/pool.v2"
)

type clientPool struct {
	mu   sync.RWMutex
	pool map[uint64]pool.Pool
}

func newClientPool() *clientPool {
	return &clientPool{
		pool: make(map[uint64]pool.Pool),
	}
}

func (c *clientPool) setPool(nodeID uint64, p pool.Pool) {
	c.mu.Lock()
	c.pool[nodeID] = p
	c.mu.Unlock()
}

func (c *clientPool) getPool(nodeID uint64) (pool.Pool, bool) {
	c.mu.RLock()
	p, ok := c.pool[nodeID]
	c.mu.RUnlock()
	return p, ok
}

func (c *clientPool) size() int {
	c.mu.RLock()
	var size int
	for _, p := range c.pool {
		size += p.Len()
	}
	c.mu.RUnlock()
	return size
}

func (c *clientPool) conn(nodeID uint64) (net.Conn, error) {
	c.mu.RLock()
	conn, err := c.pool[nodeID].Get()
	c.mu.RUnlock()
	return conn, err
}

func (c *clientPool) close() {
	c.mu.Lock()
	for _, p := range c.pool {
		p.Close()
	}
	c.mu.Unlock()
}
