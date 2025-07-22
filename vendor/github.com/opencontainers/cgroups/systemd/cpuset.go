package systemd

import (
	"errors"
	"math/big"
	"strconv"
	"strings"
)

// RangeToBits converts a text representation of a CPU mask (as written to
// or read from cgroups' cpuset.* files, e.g. "1,3-5") to a slice of bytes
// with the corresponding bits set (as consumed by systemd over dbus as
// AllowedCPUs/AllowedMemoryNodes unit property value).
func RangeToBits(str string) ([]byte, error) {
	bits := new(big.Int)

	for _, r := range strings.Split(str, ",") {
		// allow extra spaces around
		r = strings.TrimSpace(r)
		// allow empty elements (extra commas)
		if r == "" {
			continue
		}
		startr, endr, ok := strings.Cut(r, "-")
		if ok {
			start, err := strconv.ParseUint(startr, 10, 32)
			if err != nil {
				return nil, err
			}
			end, err := strconv.ParseUint(endr, 10, 32)
			if err != nil {
				return nil, err
			}
			if start > end {
				return nil, errors.New("invalid range: " + r)
			}
			for i := start; i <= end; i++ {
				bits.SetBit(bits, int(i), 1)
			}
		} else {
			val, err := strconv.ParseUint(startr, 10, 32)
			if err != nil {
				return nil, err
			}
			bits.SetBit(bits, int(val), 1)
		}
	}

	ret := bits.Bytes()
	if len(ret) == 0 {
		// do not allow empty values
		return nil, errors.New("empty value")
	}

	// fit cpuset parsing order in systemd
	for l, r := 0, len(ret)-1; l < r; l, r = l+1, r-1 {
		ret[l], ret[r] = ret[r], ret[l]
	}
	return ret, nil
}
