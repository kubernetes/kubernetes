package pq

import (
	"encoding/hex"
	"fmt"
)

// decodeUUIDBinary interprets the binary format of a uuid, returning it in text format.
func decodeUUIDBinary(src []byte) ([]byte, error) {
	if len(src) != 16 {
		return nil, fmt.Errorf("pq: unable to decode uuid; bad length: %d", len(src))
	}

	dst := make([]byte, 36)
	dst[8], dst[13], dst[18], dst[23] = '-', '-', '-', '-'
	hex.Encode(dst[0:], src[0:4])
	hex.Encode(dst[9:], src[4:6])
	hex.Encode(dst[14:], src[6:8])
	hex.Encode(dst[19:], src[8:10])
	hex.Encode(dst[24:], src[10:16])

	return dst, nil
}
