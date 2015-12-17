package cluster

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
)

// ShardMapper is responsible for providing mappers for requested shards. It is
// responsible for creating those mappers from the local store, or reaching
// out to another node on the cluster.
type ShardMapper struct {
	ForceRemoteMapping bool // All shards treated as remote. Useful for testing.

	MetaStore interface {
		NodeID() uint64
		Node(id uint64) (ni *meta.NodeInfo, err error)
	}

	TSDBStore interface {
		CreateMapper(shardID uint64, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error)
	}

	timeout time.Duration
	pool    *clientPool
}

// NewShardMapper returns a mapper of local and remote shards.
func NewShardMapper(timeout time.Duration) *ShardMapper {
	return &ShardMapper{
		pool:    newClientPool(),
		timeout: timeout,
	}
}

// CreateMapper returns a Mapper for the given shard ID.
func (s *ShardMapper) CreateMapper(sh meta.ShardInfo, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error) {
	// Create a remote mapper if the local node doesn't own the shard.
	if !sh.OwnedBy(s.MetaStore.NodeID()) || s.ForceRemoteMapping {
		// Pick a node in a pseudo-random manner.
		conn, err := s.dial(sh.Owners[rand.Intn(len(sh.Owners))].NodeID)
		if err != nil {
			return nil, err
		}
		conn.SetDeadline(time.Now().Add(s.timeout))

		return NewRemoteMapper(conn, sh.ID, stmt, chunkSize), nil
	}

	// If it is local then return the mapper from the store.
	m, err := s.TSDBStore.CreateMapper(sh.ID, stmt, chunkSize)
	if err != nil {
		return nil, err
	}

	return m, nil
}

func (s *ShardMapper) dial(nodeID uint64) (net.Conn, error) {
	ni, err := s.MetaStore.Node(nodeID)
	if err != nil {
		return nil, err
	}
	conn, err := net.Dial("tcp", ni.Host)
	if err != nil {
		return nil, err
	}

	// Write the cluster multiplexing header byte
	conn.Write([]byte{MuxHeader})

	return conn, nil
}

// RemoteMapper implements the tsdb.Mapper interface. It connects to a remote node,
// sends a query, and interprets the stream of data that comes back.
type RemoteMapper struct {
	shardID   uint64
	stmt      influxql.Statement
	chunkSize int

	tagsets []string
	fields  []string

	conn             net.Conn
	bufferedResponse *MapShardResponse

	unmarshallers []tsdb.UnmarshalFunc // Mapping-specific unmarshal functions.
}

// NewRemoteMapper returns a new remote mapper using the given connection.
func NewRemoteMapper(c net.Conn, shardID uint64, stmt influxql.Statement, chunkSize int) *RemoteMapper {
	return &RemoteMapper{
		conn:      c,
		shardID:   shardID,
		stmt:      stmt,
		chunkSize: chunkSize,
	}
}

// Open connects to the remote node and starts receiving data.
func (r *RemoteMapper) Open() (err error) {
	defer func() {
		if err != nil {
			r.conn.Close()
		}
	}()

	// Build Map request.
	var request MapShardRequest
	request.SetShardID(r.shardID)
	request.SetQuery(r.stmt.String())
	request.SetChunkSize(int32(r.chunkSize))

	// Marshal into protocol buffers.
	buf, err := request.MarshalBinary()
	if err != nil {
		return err
	}

	// Write request.
	if err := WriteTLV(r.conn, mapShardRequestMessage, buf); err != nil {
		return err
	}

	// Read the response.
	_, buf, err = ReadTLV(r.conn)
	if err != nil {
		return err
	}

	// Unmarshal response.
	r.bufferedResponse = &MapShardResponse{}
	if err := r.bufferedResponse.UnmarshalBinary(buf); err != nil {
		return err
	}

	if r.bufferedResponse.Code() != 0 {
		return fmt.Errorf("error code %d: %s", r.bufferedResponse.Code(), r.bufferedResponse.Message())
	}

	// Decode the first response to get the TagSets.
	r.tagsets = r.bufferedResponse.TagSets()
	r.fields = r.bufferedResponse.Fields()

	// Set up each mapping function for this statement.
	if stmt, ok := r.stmt.(*influxql.SelectStatement); ok {
		for _, c := range stmt.FunctionCalls() {
			fn, err := tsdb.InitializeUnmarshaller(c)
			if err != nil {
				return err
			}
			r.unmarshallers = append(r.unmarshallers, fn)
		}
	}

	return nil
}

// TagSets returns the TagSets
func (r *RemoteMapper) TagSets() []string {
	return r.tagsets
}

// Fields returns RemoteMapper's Fields
func (r *RemoteMapper) Fields() []string {
	return r.fields
}

// NextChunk returns the next chunk read from the remote node to the client.
func (r *RemoteMapper) NextChunk() (chunk interface{}, err error) {
	var response *MapShardResponse
	if r.bufferedResponse != nil {
		response = r.bufferedResponse
		r.bufferedResponse = nil
	} else {
		response = &MapShardResponse{}

		// Read the response.
		_, buf, err := ReadTLV(r.conn)
		if err != nil {
			return nil, err
		}

		// Unmarshal response.
		if err := response.UnmarshalBinary(buf); err != nil {
			return nil, err
		}

		if response.Code() != 0 {
			return nil, fmt.Errorf("error code %d: %s", response.Code(), response.Message())
		}
	}

	if response.Data() == nil {
		return nil, nil
	}

	moj := &tsdb.MapperOutputJSON{}
	if err := json.Unmarshal(response.Data(), moj); err != nil {
		return nil, err
	}
	mvj := []*tsdb.MapperValueJSON{}
	if err := json.Unmarshal(moj.Values, &mvj); err != nil {
		return nil, err
	}

	// Prep the non-JSON version of Mapper output.
	mo := &tsdb.MapperOutput{
		Name:      moj.Name,
		Tags:      moj.Tags,
		Fields:    moj.Fields,
		CursorKey: moj.CursorKey,
	}

	if len(mvj) == 1 && len(mvj[0].AggData) > 0 {
		// The MapperValue is carrying aggregate data, so run it through the
		// custom unmarshallers for the map functions through which the data
		// was mapped.
		aggValues := []interface{}{}
		for i, b := range mvj[0].AggData {
			v, err := r.unmarshallers[i](b)
			if err != nil {
				return nil, err
			}
			aggValues = append(aggValues, v)
		}
		mo.Values = []*tsdb.MapperValue{&tsdb.MapperValue{
			Time:  mvj[0].Time,
			Value: aggValues,
			Tags:  mvj[0].Tags,
		}}
	} else {
		// Must be raw data instead.
		for _, v := range mvj {
			var rawValue interface{}
			if err := json.Unmarshal(v.RawData, &rawValue); err != nil {
				return nil, err
			}

			mo.Values = append(mo.Values, &tsdb.MapperValue{
				Time:  v.Time,
				Value: rawValue,
				Tags:  v.Tags,
			})
		}
	}

	return mo, nil
}

// Close the Mapper
func (r *RemoteMapper) Close() {
	r.conn.Close()
}
