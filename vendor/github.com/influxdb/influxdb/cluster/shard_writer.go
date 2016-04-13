package cluster

import (
	"fmt"
	"net"
	"time"

	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
	"gopkg.in/fatih/pool.v2"
)

const (
	writeShardRequestMessage byte = iota + 1
	writeShardResponseMessage
)

// ShardWriter writes a set of points to a shard.
type ShardWriter struct {
	pool    *clientPool
	timeout time.Duration

	MetaStore interface {
		Node(id uint64) (ni *meta.NodeInfo, err error)
	}
}

// NewShardWriter returns a new instance of ShardWriter.
func NewShardWriter(timeout time.Duration) *ShardWriter {
	return &ShardWriter{
		pool:    newClientPool(),
		timeout: timeout,
	}
}

func (w *ShardWriter) WriteShard(shardID, ownerID uint64, points []tsdb.Point) error {
	c, err := w.dial(ownerID)
	if err != nil {
		return err
	}

	conn, ok := c.(*pool.PoolConn)
	if !ok {
		panic("wrong connection type")
	}
	defer func(conn net.Conn) {
		conn.Close() // return to pool
	}(conn)

	// Build write request.
	var request WriteShardRequest
	request.SetShardID(shardID)
	request.AddPoints(points)

	// Marshal into protocol buffers.
	buf, err := request.MarshalBinary()
	if err != nil {
		return err
	}

	// Write request.
	conn.SetWriteDeadline(time.Now().Add(w.timeout))
	if err := WriteTLV(conn, writeShardRequestMessage, buf); err != nil {
		conn.MarkUnusable()
		return err
	}

	// Read the response.
	conn.SetReadDeadline(time.Now().Add(w.timeout))
	_, buf, err = ReadTLV(conn)
	if err != nil {
		conn.MarkUnusable()
		return err
	}

	// Unmarshal response.
	var response WriteShardResponse
	if err := response.UnmarshalBinary(buf); err != nil {
		return err
	}

	if response.Code() != 0 {
		return fmt.Errorf("error code %d: %s", response.Code(), response.Message())
	}

	return nil
}

func (c *ShardWriter) dial(nodeID uint64) (net.Conn, error) {
	// If we don't have a connection pool for that addr yet, create one
	_, ok := c.pool.getPool(nodeID)
	if !ok {
		factory := &connFactory{nodeID: nodeID, clientPool: c.pool, timeout: c.timeout}
		factory.metaStore = c.MetaStore

		p, err := pool.NewChannelPool(1, 3, factory.dial)
		if err != nil {
			return nil, err
		}
		c.pool.setPool(nodeID, p)
	}
	return c.pool.conn(nodeID)
}

func (w *ShardWriter) Close() error {
	if w.pool == nil {
		return fmt.Errorf("client already closed")
	}
	w.pool.close()
	w.pool = nil
	return nil
}

const (
	maxConnections = 500
	maxRetries     = 3
)

var errMaxConnectionsExceeded = fmt.Errorf("can not exceed max connections of %d", maxConnections)

type connFactory struct {
	nodeID  uint64
	timeout time.Duration

	clientPool interface {
		size() int
	}

	metaStore interface {
		Node(id uint64) (ni *meta.NodeInfo, err error)
	}
}

func (c *connFactory) dial() (net.Conn, error) {
	if c.clientPool.size() > maxConnections {
		return nil, errMaxConnectionsExceeded
	}

	ni, err := c.metaStore.Node(c.nodeID)
	if err != nil {
		return nil, err
	}

	if ni == nil {
		return nil, fmt.Errorf("node %d does not exist", c.nodeID)
	}

	conn, err := net.DialTimeout("tcp", ni.Host, c.timeout)
	if err != nil {
		return nil, err
	}

	// Write a marker byte for cluster messages.
	_, err = conn.Write([]byte{MuxHeader})
	if err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}
