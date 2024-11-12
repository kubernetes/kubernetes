package nl

import (
	"encoding/binary"
	"fmt"
	"log"
)

type Attribute struct {
	Type  uint16
	Value []byte
}

func ParseAttributes(data []byte) <-chan Attribute {
	native := NativeEndian()
	result := make(chan Attribute)

	go func() {
		i := 0
		for i+4 < len(data) {
			length := int(native.Uint16(data[i : i+2]))
			attrType := native.Uint16(data[i+2 : i+4])

			if length < 4 {
				log.Printf("attribute 0x%02x has invalid length of %d bytes", attrType, length)
				break
			}

			if len(data) < i+length {
				log.Printf("attribute 0x%02x of length %d is truncated, only %d bytes remaining", attrType, length, len(data)-i)
				break
			}

			result <- Attribute{
				Type:  attrType,
				Value: data[i+4 : i+length],
			}
			i += rtaAlignOf(length)
		}
		close(result)
	}()

	return result
}

func PrintAttributes(data []byte) {
	printAttributes(data, 0)
}

func printAttributes(data []byte, level int) {
	for attr := range ParseAttributes(data) {
		for i := 0; i < level; i++ {
			print("> ")
		}
		nested := attr.Type&NLA_F_NESTED != 0
		fmt.Printf("type=%d nested=%v len=%v %v\n", attr.Type&NLA_TYPE_MASK, nested, len(attr.Value), attr.Value)
		if nested {
			printAttributes(attr.Value, level+1)
		}
	}
}

// Uint32 returns the uint32 value respecting the NET_BYTEORDER flag
func (attr *Attribute) Uint32() uint32 {
	if attr.Type&NLA_F_NET_BYTEORDER != 0 {
		return binary.BigEndian.Uint32(attr.Value)
	} else {
		return NativeEndian().Uint32(attr.Value)
	}
}

// Uint64 returns the uint64 value respecting the NET_BYTEORDER flag
func (attr *Attribute) Uint64() uint64 {
	if attr.Type&NLA_F_NET_BYTEORDER != 0 {
		return binary.BigEndian.Uint64(attr.Value)
	} else {
		return NativeEndian().Uint64(attr.Value)
	}
}
