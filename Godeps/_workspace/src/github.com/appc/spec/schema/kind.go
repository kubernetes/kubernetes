package schema

import (
	"encoding/json"

	"github.com/appc/spec/schema/types"
)

type Kind struct {
	ACVersion types.SemVer `json:"acVersion"`
	ACKind    types.ACKind `json:"acKind"`
}

type kind Kind

func (k *Kind) UnmarshalJSON(data []byte) error {
	nk := kind{}
	err := json.Unmarshal(data, &nk)
	if err != nil {
		return err
	}
	*k = Kind(nk)
	return nil
}

func (k Kind) MarshalJSON() ([]byte, error) {
	return json.Marshal(kind(k))
}
