package guid

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

var _ = (json.Marshaler)(&GUID{})
var _ = (json.Unmarshaler)(&GUID{})

type GUID [16]byte

func New() GUID {
	g := GUID{}
	_, err := io.ReadFull(rand.Reader, g[:])
	if err != nil {
		panic(err)
	}
	return g
}

func (g GUID) String() string {
	return fmt.Sprintf("%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x-%02x", g[3], g[2], g[1], g[0], g[5], g[4], g[7], g[6], g[8:10], g[10:])
}

func FromString(s string) GUID {
	if len(s) != 36 {
		panic(fmt.Sprintf("invalid GUID length: %d", len(s)))
	}
	if s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
		panic("invalid GUID format")
	}
	indexOrder := [16]int{
		0, 2, 4, 6,
		9, 11,
		14, 16,
		19, 21,
		24, 26, 28, 30, 32, 34,
	}
	byteOrder := [16]int{
		3, 2, 1, 0,
		5, 4,
		7, 6,
		8, 9,
		10, 11, 12, 13, 14, 15,
	}
	var g GUID
	for i, x := range indexOrder {
		b, err := strconv.ParseInt(s[x:x+2], 16, 16)
		if err != nil {
			panic(err)
		}
		g[byteOrder[i]] = byte(b)
	}
	return g
}

func (g GUID) MarshalJSON() ([]byte, error) {
	return json.Marshal(g.String())
}

func (g *GUID) UnmarshalJSON(data []byte) error {
	*g = FromString(strings.Trim(string(data), "\""))
	return nil
}
