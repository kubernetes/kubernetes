// +build !go1.7

package sign

import (
	"bytes"
	"encoding/json"
)

func encodePolicyJSON(p *Policy) ([]byte, error) {
	src, err := json.Marshal(p)
	// Convert \u0026 back to &
	return bytes.Replace(src, []byte("\\u0026"), []byte("&"), -1), err
}
