package dns

import (
	"net"
	"reflect"
	"strconv"
)

// NumField returns the number of rdata fields r has.
func NumField(r RR) int {
	return reflect.ValueOf(r).Elem().NumField() - 1 // Remove RR_Header
}

// Field returns the rdata field i as a string. Fields are indexed starting from 1.
// RR types that holds slice data, for instance the NSEC type bitmap will return a single
// string where the types are concatenated using a space.
// Accessing non existing fields will cause a panic.
func Field(r RR, i int) string {
	if i == 0 {
		return ""
	}
	d := reflect.ValueOf(r).Elem().Field(i)
	switch k := d.Kind(); k {
	case reflect.String:
		return d.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(d.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return strconv.FormatUint(d.Uint(), 10)
	case reflect.Slice:
		switch reflect.ValueOf(r).Elem().Type().Field(i).Tag {
		case `dns:"a"`:
			// TODO(miek): Hmm store this as 16 bytes
			if d.Len() < net.IPv6len {
				return net.IPv4(byte(d.Index(0).Uint()),
					byte(d.Index(1).Uint()),
					byte(d.Index(2).Uint()),
					byte(d.Index(3).Uint())).String()
			}
			return net.IPv4(byte(d.Index(12).Uint()),
				byte(d.Index(13).Uint()),
				byte(d.Index(14).Uint()),
				byte(d.Index(15).Uint())).String()
		case `dns:"aaaa"`:
			return net.IP{
				byte(d.Index(0).Uint()),
				byte(d.Index(1).Uint()),
				byte(d.Index(2).Uint()),
				byte(d.Index(3).Uint()),
				byte(d.Index(4).Uint()),
				byte(d.Index(5).Uint()),
				byte(d.Index(6).Uint()),
				byte(d.Index(7).Uint()),
				byte(d.Index(8).Uint()),
				byte(d.Index(9).Uint()),
				byte(d.Index(10).Uint()),
				byte(d.Index(11).Uint()),
				byte(d.Index(12).Uint()),
				byte(d.Index(13).Uint()),
				byte(d.Index(14).Uint()),
				byte(d.Index(15).Uint()),
			}.String()
		case `dns:"nsec"`:
			if d.Len() == 0 {
				return ""
			}
			s := Type(d.Index(0).Uint()).String()
			for i := 1; i < d.Len(); i++ {
				s += " " + Type(d.Index(i).Uint()).String()
			}
			return s
		case `dns:"wks"`:
			if d.Len() == 0 {
				return ""
			}
			s := strconv.Itoa(int(d.Index(0).Uint()))
			for i := 0; i < d.Len(); i++ {
				s += " " + strconv.Itoa(int(d.Index(i).Uint()))
			}
			return s
		default:
			// if it does not have a tag its a string slice
			fallthrough
		case `dns:"txt"`:
			if d.Len() == 0 {
				return ""
			}
			s := d.Index(0).String()
			for i := 1; i < d.Len(); i++ {
				s += " " + d.Index(i).String()
			}
			return s
		}
	}
	return ""
}
