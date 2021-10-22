package issue260

import (
	"encoding/json"
	"time"

	"github.com/gogo/protobuf/jsonpb"
)

type Dropped struct {
	Name string
	Age  int32
}

func (d *Dropped) UnmarshalJSONPB(u *jsonpb.Unmarshaler, b []byte) error {
	return json.Unmarshal(b, d)
}

func (d *Dropped) MarshalJSONPB(*jsonpb.Marshaler) ([]byte, error) {
	return json.Marshal(d)
}

func (d *Dropped) Drop() bool {
	return true
}

type DroppedWithoutGetters struct {
	Width             int64
	Height            int64
	Timestamp         time.Time  `protobuf:"bytes,3,opt,name=timestamp,stdtime" json:"timestamp"`
	NullableTimestamp *time.Time `protobuf:"bytes,4,opt,name=nullable_timestamp,json=nullableTimestamp,stdtime" json:"nullable_timestamp,omitempty"`
}

func (d *DroppedWithoutGetters) UnmarshalJSONPB(u *jsonpb.Unmarshaler, b []byte) error {
	return json.Unmarshal(b, d)
}

func (d *DroppedWithoutGetters) MarshalJSONPB(*jsonpb.Marshaler) ([]byte, error) {
	return json.Marshal(d)
}
