package cluster

import (
	"fmt"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/influxdb/influxdb/cluster/internal"
	"github.com/influxdb/influxdb/models"
)

//go:generate protoc --gogo_out=. internal/data.proto

// MapShardRequest represents the request to map a remote shard for a query.
type MapShardRequest struct {
	pb internal.MapShardRequest
}

// ShardID of the map request
func (m *MapShardRequest) ShardID() uint64 { return m.pb.GetShardID() }

// Query returns the Shard map request's query
func (m *MapShardRequest) Query() string { return m.pb.GetQuery() }

// ChunkSize returns Shard map request's chunk size
func (m *MapShardRequest) ChunkSize() int32 { return m.pb.GetChunkSize() }

// SetShardID sets the map request's shard id
func (m *MapShardRequest) SetShardID(id uint64) { m.pb.ShardID = &id }

// SetQuery sets the Shard map request's Query
func (m *MapShardRequest) SetQuery(query string) { m.pb.Query = &query }

// SetChunkSize sets the Shard map request's chunk size
func (m *MapShardRequest) SetChunkSize(chunkSize int32) { m.pb.ChunkSize = &chunkSize }

// MarshalBinary encodes the object to a binary format.
func (m *MapShardRequest) MarshalBinary() ([]byte, error) {
	return proto.Marshal(&m.pb)
}

// UnmarshalBinary populates MapShardRequest from a binary format.
func (m *MapShardRequest) UnmarshalBinary(buf []byte) error {
	if err := proto.Unmarshal(buf, &m.pb); err != nil {
		return err
	}
	return nil
}

// MapShardResponse represents the response returned from a remote MapShardRequest call
type MapShardResponse struct {
	pb internal.MapShardResponse
}

// NewMapShardResponse returns the response returned from a remote MapShardRequest call
func NewMapShardResponse(code int, message string) *MapShardResponse {
	m := &MapShardResponse{}
	m.SetCode(code)
	m.SetMessage(message)
	return m
}

// Code returns the Shard map response's code
func (r *MapShardResponse) Code() int { return int(r.pb.GetCode()) }

// Message returns the the Shard map response's Message
func (r *MapShardResponse) Message() string { return r.pb.GetMessage() }

// TagSets returns Shard map response's tag sets
func (r *MapShardResponse) TagSets() []string { return r.pb.GetTagSets() }

// Fields returns the Shard map response's Fields
func (r *MapShardResponse) Fields() []string { return r.pb.GetFields() }

// Data returns the Shard map response's Data
func (r *MapShardResponse) Data() []byte { return r.pb.GetData() }

// SetCode sets the Shard map response's code
func (r *MapShardResponse) SetCode(code int) { r.pb.Code = proto.Int32(int32(code)) }

// SetMessage sets Shard map response's message
func (r *MapShardResponse) SetMessage(message string) { r.pb.Message = &message }

// SetTagSets sets Shard map response's tagsets
func (r *MapShardResponse) SetTagSets(tagsets []string) { r.pb.TagSets = tagsets }

// SetFields sets the Shard map response's Fields
func (r *MapShardResponse) SetFields(fields []string) { r.pb.Fields = fields }

// SetData sets the Shard map response's Data
func (r *MapShardResponse) SetData(data []byte) { r.pb.Data = data }

// MarshalBinary encodes the object to a binary format.
func (r *MapShardResponse) MarshalBinary() ([]byte, error) {
	return proto.Marshal(&r.pb)
}

// UnmarshalBinary populates WritePointRequest from a binary format.
func (r *MapShardResponse) UnmarshalBinary(buf []byte) error {
	if err := proto.Unmarshal(buf, &r.pb); err != nil {
		return err
	}
	return nil
}

// WritePointsRequest represents a request to write point data to the cluster
type WritePointsRequest struct {
	Database         string
	RetentionPolicy  string
	ConsistencyLevel ConsistencyLevel
	Points           []models.Point
}

// AddPoint adds a point to the WritePointRequest with field key 'value'
func (w *WritePointsRequest) AddPoint(name string, value interface{}, timestamp time.Time, tags map[string]string) {
	pt, err := models.NewPoint(
		name, tags, map[string]interface{}{"value": value}, timestamp,
	)
	if err != nil {
		return
	}
	w.Points = append(w.Points, pt)
}

// WriteShardRequest represents the a request to write a slice of points to a shard
type WriteShardRequest struct {
	pb internal.WriteShardRequest
}

// WriteShardResponse represents the response returned from a remote WriteShardRequest call
type WriteShardResponse struct {
	pb internal.WriteShardResponse
}

// SetShardID sets the ShardID
func (w *WriteShardRequest) SetShardID(id uint64) { w.pb.ShardID = &id }

// ShardID gets the ShardID
func (w *WriteShardRequest) ShardID() uint64 { return w.pb.GetShardID() }

// Points returns the time series Points
func (w *WriteShardRequest) Points() []models.Point { return w.unmarshalPoints() }

// AddPoint adds a new time series point
func (w *WriteShardRequest) AddPoint(name string, value interface{}, timestamp time.Time, tags map[string]string) {
	pt, err := models.NewPoint(
		name, tags, map[string]interface{}{"value": value}, timestamp,
	)
	if err != nil {
		return
	}
	w.AddPoints([]models.Point{pt})
}

// AddPoints adds a new time series point
func (w *WriteShardRequest) AddPoints(points []models.Point) {
	for _, p := range points {
		w.pb.Points = append(w.pb.Points, []byte(p.String()))
	}
}

// MarshalBinary encodes the object to a binary format.
func (w *WriteShardRequest) MarshalBinary() ([]byte, error) {
	return proto.Marshal(&w.pb)
}

// UnmarshalBinary populates WritePointRequest from a binary format.
func (w *WriteShardRequest) UnmarshalBinary(buf []byte) error {
	if err := proto.Unmarshal(buf, &w.pb); err != nil {
		return err
	}
	return nil
}

func (w *WriteShardRequest) unmarshalPoints() []models.Point {
	points := make([]models.Point, len(w.pb.GetPoints()))
	for i, p := range w.pb.GetPoints() {
		pt, err := models.ParsePoints(p)
		if err != nil {
			// A error here means that one node parsed the point correctly but sent an
			// unparseable version to another node.  We could log and drop the point and allow
			// anti-entropy to resolve the discrepancy but this shouldn't ever happen.
			panic(fmt.Sprintf("failed to parse point: `%v`: %v", string(p), err))
		}
		points[i] = pt[0]
	}
	return points
}

// SetCode sets the Code
func (w *WriteShardResponse) SetCode(code int) { w.pb.Code = proto.Int32(int32(code)) }

// SetMessage sets the Message
func (w *WriteShardResponse) SetMessage(message string) { w.pb.Message = &message }

// Code returns the Code
func (w *WriteShardResponse) Code() int { return int(w.pb.GetCode()) }

// Message returns the Message
func (w *WriteShardResponse) Message() string { return w.pb.GetMessage() }

// MarshalBinary encodes the object to a binary format.
func (w *WriteShardResponse) MarshalBinary() ([]byte, error) {
	return proto.Marshal(&w.pb)
}

// UnmarshalBinary populates WritePointRequest from a binary format.
func (w *WriteShardResponse) UnmarshalBinary(buf []byte) error {
	if err := proto.Unmarshal(buf, &w.pb); err != nil {
		return err
	}
	return nil
}
