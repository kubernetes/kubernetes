package test

import (
	"encoding/json"
	"strings"

	"github.com/gogo/protobuf/proto"
)

type T struct {
	Data string
}

func (gt *T) protoType() *ProtoType {
	return &ProtoType{
		Field2: &gt.Data,
	}
}

func (gt T) Equal(other T) bool {
	return gt.protoType().Equal(other.protoType())
}

func (gt *T) Size() int {
	proto := &ProtoType{
		Field2: &gt.Data,
	}
	return proto.Size()
}

func NewPopulatedT(r randyThetest) *T {
	data := NewPopulatedProtoType(r, false).Field2
	gt := &T{}
	if data != nil {
		gt.Data = *data
	}
	return gt
}

func (r T) Marshal() ([]byte, error) {
	return proto.Marshal(r.protoType())
}

func (r *T) MarshalTo(data []byte) (n int, err error) {
	return r.protoType().MarshalTo(data)
}

func (r *T) Unmarshal(data []byte) error {
	pr := &ProtoType{}
	err := proto.Unmarshal(data, pr)
	if err != nil {
		return err
	}

	if pr.Field2 != nil {
		r.Data = *pr.Field2
	}
	return nil
}

func (gt T) MarshalJSON() ([]byte, error) {
	return json.Marshal(gt.Data)
}

func (gt *T) UnmarshalJSON(data []byte) error {
	var s string
	err := json.Unmarshal(data, &s)
	if err != nil {
		return err
	}
	*gt = T{Data: s}
	return nil
}

func (gt T) Compare(other T) int {
	return strings.Compare(gt.Data, other.Data)
}
