package s3crypto

import (
	"encoding/json"
)

// MaterialDescription is used to identify how and what master
// key has been used.
type MaterialDescription map[string]*string

func (md *MaterialDescription) encodeDescription() ([]byte, error) {
	v, err := json.Marshal(&md)
	return v, err
}

func (md *MaterialDescription) decodeDescription(b []byte) error {
	return json.Unmarshal(b, &md)
}
