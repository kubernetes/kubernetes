package nlgo

import (
	"bytes"
	"syscall"
	"testing"
)

func TestSimple(t *testing.T) {
	policy := MapPolicy{
		Prefix: "T",
		Names: map[uint16]string{
			1: "U8",
			2: "U16",
			3: "U32",
			4: "U64",
			5: "String",
		},
		Rule: map[uint16]Policy{
			1: U8Policy,
			2: U16Policy,
			3: U32Policy,
			4: U64Policy,
			5: StringPolicy,
		},
	}

	attrs := AttrMap{
		Policy: policy,
		AttrSlice: AttrSlice{
			Attr{
				Header: syscall.NlAttr{
					Type: 1,
				},
				Value: U8(1),
			},
			Attr{
				Header: syscall.NlAttr{
					Type: 2,
				},
				Value: U16(2),
			},
			Attr{
				Header: syscall.NlAttr{
					Type: 3,
				},
				Value: U32(3),
			},
			Attr{
				Header: syscall.NlAttr{
					Type: 4,
				},
				Value: U64(4),
			},
			Attr{
				Header: syscall.NlAttr{
					Type: 5,
				},
				Value: String("5"),
			},
		},
	}
	buf := attrs.Bytes()
	if attr, err := policy.Parse(buf); err != nil {
		t.Error(err)
	} else if amap, ok := attr.(AttrMap); !ok {
		t.Error("policy does not return AttrMap")
	} else if !bytes.Equal(buf, amap.Bytes()) {
		t.Error("cycle error")
	}
}
