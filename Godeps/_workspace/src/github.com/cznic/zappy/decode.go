// Copyright 2014 The zappy Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the SNAPPY-GO-LICENSE file.

package zappy

import (
	"encoding/binary"
	"errors"
)

// ErrCorrupt reports that the input is invalid.
var ErrCorrupt = errors.New("zappy: corrupt input")

// DecodedLen returns the length of the decoded block.
func DecodedLen(src []byte) (int, error) {
	v, _, err := decodedLen(src)
	return v, err
}

// decodedLen returns the length of the decoded block and the number of bytes
// that the length header occupied.
func decodedLen(src []byte) (blockLen, headerLen int, err error) {
	v, n := binary.Uvarint(src)
	if n == 0 {
		return 0, 0, ErrCorrupt
	}

	if uint64(int(v)) != v {
		return 0, 0, errors.New("zappy: decoded block is too large")
	}

	return int(v), n, nil
}
