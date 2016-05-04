package dns

// These raw* functions do not use reflection, they directly set the values
// in the buffer. There are faster than their reflection counterparts.

// RawSetId sets the message id in buf.
func rawSetId(msg []byte, i uint16) bool {
	if len(msg) < 2 {
		return false
	}
	msg[0], msg[1] = packUint16(i)
	return true
}

// rawSetQuestionLen sets the length of the question section.
func rawSetQuestionLen(msg []byte, i uint16) bool {
	if len(msg) < 6 {
		return false
	}
	msg[4], msg[5] = packUint16(i)
	return true
}

// rawSetAnswerLen sets the length of the answer section.
func rawSetAnswerLen(msg []byte, i uint16) bool {
	if len(msg) < 8 {
		return false
	}
	msg[6], msg[7] = packUint16(i)
	return true
}

// rawSetsNsLen sets the length of the authority section.
func rawSetNsLen(msg []byte, i uint16) bool {
	if len(msg) < 10 {
		return false
	}
	msg[8], msg[9] = packUint16(i)
	return true
}

// rawSetExtraLen sets the length of the additional section.
func rawSetExtraLen(msg []byte, i uint16) bool {
	if len(msg) < 12 {
		return false
	}
	msg[10], msg[11] = packUint16(i)
	return true
}

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
	msg[off], msg[off+1] = packUint16(uint16(rdatalen))
	return true
}
