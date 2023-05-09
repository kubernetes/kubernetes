package systemd

import (
	"encoding/binary"
	"strconv"
	"strings"

	"github.com/bits-and-blooms/bitset"
	"github.com/pkg/errors"
)

// RangeToBits converts a text representation of a CPU mask (as written to
// or read from cgroups' cpuset.* files, e.g. "1,3-5") to a slice of bytes
// with the corresponding bits set (as consumed by systemd over dbus as
// AllowedCPUs/AllowedMemoryNodes unit property value).
func RangeToBits(str string) ([]byte, error) {
	bits := &bitset.BitSet{}

	for _, r := range strings.Split(str, ",") {
		// allow extra spaces around
		r = strings.TrimSpace(r)
		// allow empty elements (extra commas)
		if r == "" {
			continue
		}
		ranges := strings.SplitN(r, "-", 2)
		if len(ranges) > 1 {
			start, err := strconv.ParseUint(ranges[0], 10, 32)
			if err != nil {
				return nil, err
			}
			end, err := strconv.ParseUint(ranges[1], 10, 32)
			if err != nil {
				return nil, err
			}
			if start > end {
				return nil, errors.New("invalid range: " + r)
			}
			for i := uint(start); i <= uint(end); i++ {
				bits.Set(i)
			}
		} else {
			val, err := strconv.ParseUint(ranges[0], 10, 32)
			if err != nil {
				return nil, err
			}
			bits.Set(uint(val))
		}
	}

	val := bits.Bytes()
	if len(val) == 0 {
		// do not allow empty values
		return nil, errors.New("empty value")
	}

	// fit cpuset parsing order in systemd
	for l, r := 0, len(ret)-1; l < r; l, r = l+1, r-1 {
		ret[l], ret[r] = ret[r], ret[l]
	}
	return ret, nil
}
