package misspell

// ByteToUpper converts an ascii byte to upper cases
// Uses a branchless algorithm
func ByteToUpper(x byte) byte {
	b := byte(0x80) | x
	c := b - byte(0x61)
	d := ^(b - byte(0x7b))
	e := (c & d) & (^x & 0x7f)
	return x - (e >> 2)
}

// ByteToLower converts an ascii byte to lower case
// uses a branchless algorithm
func ByteToLower(eax byte) byte {
	ebx := eax&byte(0x7f) + byte(0x25)
	ebx = ebx&byte(0x7f) + byte(0x1a)
	ebx = ((ebx & ^eax) >> 2) & byte(0x20)
	return eax + ebx
}

// ByteEqualFold does ascii compare, case insensitive
func ByteEqualFold(a, b byte) bool {
	return a == b || ByteToLower(a) == ByteToLower(b)
}

// StringEqualFold ASCII case-insensitive comparison
// golang toUpper/toLower for both bytes and strings
// appears to be Unicode based which is super slow
// based from https://codereview.appspot.com/5180044/patch/14007/21002
func StringEqualFold(s1, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := 0; i < len(s1); i++ {
		c1 := s1[i]
		c2 := s2[i]
		// c1 & c2
		if c1 != c2 {
			c1 |= 'a' - 'A'
			c2 |= 'a' - 'A'
			if c1 != c2 || c1 < 'a' || c1 > 'z' {
				return false
			}
		}
	}
	return true
}

// StringHasPrefixFold is similar to strings.HasPrefix but comparison
// is done ignoring ASCII case.
// /
func StringHasPrefixFold(s1, s2 string) bool {
	// prefix is bigger than input --> false
	if len(s1) < len(s2) {
		return false
	}
	if len(s1) == len(s2) {
		return StringEqualFold(s1, s2)
	}
	return StringEqualFold(s1[:len(s2)], s2)
}
