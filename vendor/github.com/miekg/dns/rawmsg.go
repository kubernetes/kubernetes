package dns

import "encoding/binary"

// rawSetRdlength sets the rdlength in the header of
// the RR. The offset 'off' must be positioned at the
// start of the header of the RR, 'end' must be the
// end of the RR.
func rawSetRdlength(msg []byte, off, end int) bool {
	l := len(msg)
Loop:
	for {
		if off+1 > l {
			return false
		}
		c := int(msg[off])
		off++
		switch c & 0xC0 {
		case 0x00:
			if c == 0x00 {
				// End of the domainname
				break Loop
			}
			if off+c > l {
				return false
			}
			off += c

		case 0xC0:
			// pointer, next byte included, ends domainname
			off++
			break Loop
		}
	}
	// The domainname has been seen, we at the start of the fixed part in the header.
	// Type is 2 bytes, class is 2 bytes, ttl 4 and then 2 bytes for the length.
	off += 2 + 2 + 4
	if off+2 > l {
		return false
	}
	//off+1 is the end of the header, 'end' is the end of the rr
	//so 'end' - 'off+2' is the length of the rdata
	rdatalen := end - (off + 2)
	if rdatalen > 0xFFFF {
		return false
	}
	binary.BigEndian.PutUint16(msg[off:], uint16(rdatalen))
	return true
}
