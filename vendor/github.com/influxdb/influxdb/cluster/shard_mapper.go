package cluster

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"

	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
)

const (
	MAX_MAP_RESPONSE_SIZE = 1024 * 1024 * 1024
)

// ShardMapper is responsible for providing mappers for requested shards. It is
// responsible for creating those mappers from the local store, or reaching
// out to another node on the cluster.
type ShardMapper struct {
	MetaStore interface {
		NodeID() uint64
		Node(id uint64) (ni *meta.NodeInfo, err error)
	}

	TSDBStore interface {
		CreateMapper(shardID uint64, query string, chunkSize int) (tsdb.Mapper, error)
	}
}

// NewShardMapper returns a mapper of local and remote shards.
func NewShardMapper() *ShardMapper {
	return &ShardMapper{}
}

// CreateMapper returns a Mapper for the given shard ID.
func (r *ShardMapper) CreateMapper(sh meta.ShardInfo, stmt string, chunkSize int) (tsdb.Mapper, error) {
	var err error
	var m tsdb.Mapper
	if sh.OwnedBy(r.MetaStore.NodeID()) {
		m, err = r.TSDBStore.CreateMapper(sh.ID, stmt, chunkSize)
		if err != nil {
			return nil, err
		}
	} else {
		rm := NewRemoteMaper(sh.OwnerIDs[0], sh.ID, stmt, chunkSize)
		rm.MetaStore = r.MetaStore
		m = rm
	}

	return m, nil
}

// RemoteMapper implements the tsdb.Mapper interface. It connects to a remote node,
// sends a query, and interprets the stream of data that comes back.
type RemoteMapper struct {
	MetaStore interface {
		Node(id uint64) (ni *meta.NodeInfo, err error)
	}

	nodeID    uint64
	shardID   uint64
	stmt      string
	chunkSize int

	tagsets []string

	resp    *http.Response
	decoder *json.Decoder

	buffer     tsdb.MapperResponse
	bufferSent bool
}

// NewRemoteMaper returns a new remote mapper.
func NewRemoteMaper(nodeID, shardID uint64, stmt string, chunkSize int) *RemoteMapper {
	return &RemoteMapper{
		nodeID:    nodeID,
		shardID:   shardID,
		stmt:      stmt,
		chunkSize: chunkSize,
	}
}

// Open connects to the remote node and starts receiving data.
func (r *RemoteMapper) Open() error {
	node, err := r.MetaStore.Node(r.nodeID)
	if err != nil {
		return err
	}

	v := url.Values{}
	v.Set("shard", string(r.shardID))
	v.Set("q", r.stmt)
	if r.chunkSize != 0 {
		v.Set("chunksize", strconv.Itoa(r.chunkSize))
	}

	u := url.URL{
		Scheme:   "http",
		Host:     node.Host,
		RawQuery: v.Encode(),
		Path:     "/shard_mapping",
	}

	resp, err := http.Get(u.String())
	if err != nil {
		return err
	}
	r.resp = resp

	// Set up the decoder.
	lr := io.LimitReader(r.resp.Body, MAX_MAP_RESPONSE_SIZE)
	r.decoder = json.NewDecoder(lr)

	// Decode the first response to get the TagSets.
	err = r.decoder.Decode(&r.buffer)
	if err != nil {
		return err
	}
	r.tagsets = r.buffer.TagSets

	return nil
}

func (r *RemoteMapper) TagSets() []string {
	return r.tagsets
}

// NextChunk returns the next chunk read from the remote node to the client.
func (r *RemoteMapper) NextChunk() (interface{}, error) {
	if !r.bufferSent {
		r.bufferSent = true
		return r.buffer.Data, nil
	}

	mr := tsdb.MapperResponse{}
	err := r.decoder.Decode(&mr)
	if err != nil {
		return nil, err
	}

	return mr.Data, nil
}

// Close the response body
func (r *RemoteMapper) Close() {
	if r.resp != nil && r.resp.Body != nil {
		r.resp.Body.Close()
	}
}
