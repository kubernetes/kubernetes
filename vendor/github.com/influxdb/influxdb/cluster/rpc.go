package cluster

import (
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/influxdb/influxdb/cluster/internal"
	"github.com/influxdb/influxdb/tsdb"
)

//go:generate protoc --gogo_out=. internal/data.proto

// WritePointsRequest represents a request to write point data to the cluster
type WritePointsRequest struct {
	Database         string
	RetentionPolicy  string
	ConsistencyLevel ConsistencyLevel
	Points           []tsdb.Point
}

// AddPoint adds a point to the WritePointRequest with field name 'value'
func (w *WritePointsRequest) AddPoint(name string, value interface{}, timestamp time.Time, tags map[string]string) {
	w.Points = append(w.Points, tsdb.NewPoint(
		name, tags, map[string]interface{}{"value": value}, timestamp,
	))
}

// WriteShardRequest represents the a request to write a slice of points to a shard
type WriteShardRequest struct {
	pb internal.WriteShardRequest
}

// WriteShardResponse represents the response returned from a remote WriteShardRequest call
type WriteShardResponse struct {
	pb internal.WriteShardResponse
}

func (w *WriteShardRequest) SetShardID(id uint64) { w.pb.ShardID = &id }
func (w *WriteShardRequest) ShardID() uint64      { return w.pb.GetShardID() }

func (w *WriteShardRequest) Points() []tsdb.Point { return w.unmarshalPoints() }

func (w *WriteShardRequest) AddPoint(name string, value interface{}, timestamp time.Time, tags map[string]string) {
	w.AddPoints([]tsdb.Point{tsdb.NewPoint(
		name, tags, map[string]interface{}{"value": value}, timestamp,
	)})
}

func (w *WriteShardRequest) AddPoints(points []tsdb.Point) {
	w.pb.Points = append(w.pb.Points, w.marshalPoints(points)...)
}

// MarshalBinary encodes the object to a binary format.
func (w *WriteShardRequest) MarshalBinary() ([]byte, error) {
	return proto.Marshal(&w.pb)
}

func (w *WriteShardRequest) marshalPoints(points []tsdb.Point) []*internal.Point {
	pts := make([]*internal.Point, len(points))
	for i, p := range points {
		fields := []*internal.Field{}
		for k, v := range p.Fields() {
			name := k
			f := &internal.Field{
				Name: &name,
			}
			switch t := v.(type) {
			case int:
				f.Int64 = proto.Int64(int64(t))
			case int32:
				f.Int32 = proto.Int32(t)
			case int64:
				f.Int64 = proto.Int64(t)
			case float64:
				f.Float64 = proto.Float64(t)
			case bool:
				f.Bool = proto.Bool(t)
			case string:
				f.String_ = proto.String(t)
			case []byte:
				f.Bytes = t
			}
			fields = append(fields, f)
		}

		tags := []*internal.Tag{}
		for k, v := range p.Tags() {
			key := k
			value := v
			tags = append(tags, &internal.Tag{
				Key:   &key,
				Value: &value,
			})
		}
		name := p.Name()
		pts[i] = &internal.Point{
			Name:   &name,
			Time:   proto.Int64(p.Time().UnixNano()),
			Fields: fields,
			Tags:   tags,
		}

	}
	return pts
}

// UnmarshalBinary populates WritePointRequest from a binary format.
func (w *WriteShardRequest) UnmarshalBinary(buf []byte) error {
	if err := proto.Unmarshal(buf, &w.pb); err != nil {
		return err
	}
	return nil
}

func (w *WriteShardRequest) unmarshalPoints() []tsdb.Point {
	points := make([]tsdb.Point, len(w.pb.GetPoints()))
	for i, p := range w.pb.GetPoints() {
		pt := tsdb.NewPoint(
			p.GetName(), map[string]string{},
			map[string]interface{}{}, time.Unix(0, p.GetTime()))

		for _, f := range p.GetFields() {
			n := f.GetName()
			if f.Int32 != nil {
				pt.AddField(n, f.GetInt32())
			} else if f.Int64 != nil {
				pt.AddField(n, f.GetInt64())
			} else if f.Float64 != nil {
				pt.AddField(n, f.GetFloat64())
			} else if f.Bool != nil {
				pt.AddField(n, f.GetBool())
			} else if f.String_ != nil {
				pt.AddField(n, f.GetString_())
			} else {
				pt.AddField(n, f.GetBytes())
			}
		}

		tags := tsdb.Tags{}
		for _, t := range p.GetTags() {
			tags[t.GetKey()] = t.GetValue()
		}
		pt.SetTags(tags)
		points[i] = pt
	}
	return points
}

func (w *WriteShardResponse) SetCode(code int)          { w.pb.Code = proto.Int32(int32(code)) }
func (w *WriteShardResponse) SetMessage(message string) { w.pb.Message = &message }

func (w *WriteShardResponse) Code() int       { return int(w.pb.GetCode()) }
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
