// +build go1.7

package sign

import (
	"bytes"
	"encoding/json"
)

func encodePolicyJSON(p *Policy) ([]byte, error) {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(p)
	return buffer.Bytes(), err
}
