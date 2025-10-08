package imported

import (
	"encoding/json"

	"github.com/gogo/protobuf/proto"
)

type B struct {
	A
}

func (b B) Equal(other B) bool {
	return b.A.Equal(other.A)
}

func (b B) Size() int {
	return b.A.Size()
}

func NewPopulatedB(r randyA) *B {
	a := NewPopulatedA(r, true)
	if a == nil {
		return nil
	}
	return &B{*a}
}

func (b B) Marshal() ([]byte, error) {
	return proto.Marshal(&b.A)
}

func (b *B) Unmarshal(data []byte) error {
	a := &A{}
	err := proto.Unmarshal(data, a)
	if err != nil {
		return err
	}
	b.A = *a
	return nil
}

func (b B) MarshalJSON() ([]byte, error) {
	return json.Marshal(b.A)
}

func (b *B) UnmarshalJSON(data []byte) error {
	a := &A{}
	err := json.Unmarshal(data, a)
	if err != nil {
		return err
	}
	*b = B{A: *a}
	return nil
}
