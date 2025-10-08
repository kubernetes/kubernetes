package typedeclall

import (
	"encoding/json"

	"github.com/gogo/protobuf/jsonpb"
)

type Dropped struct {
	Name string
	Age  int32
}

func (d *Dropped) Drop() bool {
	return true
}

func (d *Dropped) UnmarshalJSONPB(u *jsonpb.Unmarshaler, b []byte) error {
	return json.Unmarshal(b, d)
}

func (d *Dropped) MarshalJSONPB(*jsonpb.Marshaler) ([]byte, error) {
	return json.Marshal(d)
}

type DroppedWithoutGetters struct {
	Width  int64
	Height int64
}

func (d *DroppedWithoutGetters) GetHeight() int64 {
	return d.Height
}

func (d *DroppedWithoutGetters) UnmarshalJSONPB(u *jsonpb.Unmarshaler, b []byte) error {
	return json.Unmarshal(b, d)
}

func (d *DroppedWithoutGetters) MarshalJSONPB(*jsonpb.Marshaler) ([]byte, error) {
	return json.Marshal(d)
}
