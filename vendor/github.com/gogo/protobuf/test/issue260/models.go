package issue260

import "time"

type Dropped struct {
	Name string
	Age  int32
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
