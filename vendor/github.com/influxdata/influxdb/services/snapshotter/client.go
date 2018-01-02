package snapshotter

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/tcp"
)

// Client provides an API for the snapshotter service.
type Client struct {
	host string
}

// NewClient returns a new *Client.
func NewClient(host string) *Client {
	return &Client{host: host}
}

// MetastoreBackup returns a snapshot of the meta store.
func (c *Client) MetastoreBackup() (*meta.Data, error) {
	req := &Request{
		Type: RequestMetastoreBackup,
	}

	b, err := c.doRequest(req)
	if err != nil {
		return nil, err
	}

	// Check the magic.
	magic := binary.BigEndian.Uint64(b[:8])
	if magic != BackupMagicHeader {
		return nil, errors.New("invalid metadata received")
	}
	i := 8

	// Size of the meta store bytes.
	length := int(binary.BigEndian.Uint64(b[i : i+8]))
	i += 8
	metaBytes := b[i : i+length]
	i += int(length)

	// Unpack meta data.
	var data meta.Data
	if err := data.UnmarshalBinary(metaBytes); err != nil {
		return nil, fmt.Errorf("unmarshal: %s", err)
	}

	return &data, nil
}

// doRequest sends a request to the snapshotter service and returns the result.
func (c *Client) doRequest(req *Request) ([]byte, error) {
	// Connect to snapshotter service.
	conn, err := tcp.Dial("tcp", c.host, MuxHeader)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	// Write the request
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return nil, fmt.Errorf("encode snapshot request: %s", err)
	}

	// Read snapshot from the connection
	var buf bytes.Buffer
	_, err = io.Copy(&buf, conn)

	return buf.Bytes(), err
}
