package bindata

// fmtInt formats v into the tail of buf.
// It returns the index where the output begins.
func fmtInt(buf []byte, v uint64) int {
	w := len(buf)
	if v == 0 {
		w--
		buf[w] = '0'
	} else {
		for v > 0 {
			w--
			buf[w] = byte(v%10) + '0'
			v /= 10
		}
	}
	return w
}

// fmtFrac formats the fraction of v/10**prec (e.g., ".12345") into the
// tail of buf, omitting trailing zeros. it omits the decimal
// point too when the fraction is 0. It returns the index where the
// output bytes begin and the value v/10**prec.
func fmtFrac(buf []byte, v uint64, prec int) (nw int, nv uint64) {
	// Omit trailing zeros up to and including decimal point.
	w := len(buf)
	print := false
	for i := 0; i < prec; i++ {
		digit := v % 10
		print = print || digit != 0
		if print {
			w--
			buf[w] = byte(digit) + '0'
		}
		v /= 10
	}
	if print {
		w--
		buf[w] = '.'
	}
	return w, v
}

const (
	bit   bits = 1
	byte_      = 8 * bit
	// https://en.wikipedia.org/wiki/Orders_of_magnitude_(data)
	kilobyte = 1000 * byte_
	megabyte = 1000 * kilobyte
	gigabyte = 1000 * megabyte
	terabyte = 1000 * gigabyte
	petabyte = 1000 * terabyte
	exabyte  = 1000 * petabyte
)

// Bits represents a quantity of bits, bytes, kilobytes or megabytes. Bits are
// parsed and formatted using the IEEE / SI standards, which use multiples of
// 1000 to represent kilobytes and megabytes (instead of multiples of 1024). For
// more information see https://en.wikipedia.org/wiki/Megabyte#Definitions.
type bits int64

// Bytes returns the size as a floating point number of bytes.
func (b bits) Bytes() float64 {
	bytes := b / byte_
	bits := b % byte_
	return float64(bytes) + float64(bits)/8
}

// Kilobytes returns the size as a floating point number of kilobytes.
func (b bits) Kilobytes() float64 {
	bytes := b / kilobyte
	bits := b % kilobyte
	return float64(bytes) + float64(bits)/(8*1000)
}

// Megabytes returns the size as a floating point number of megabytes.
func (b bits) Megabytes() float64 {
	bytes := b / megabyte
	bits := b % megabyte
	return float64(bytes) + float64(bits)/(8*1000*1000)
}

// Gigabytes returns the size as a floating point number of gigabytes.
func (b bits) Gigabytes() float64 {
	bytes := b / gigabyte
	bits := b % gigabyte
	return float64(bytes) + float64(bits)/(8*1000*1000*1000)
}

// String returns a string representation of b using the largest unit that has a
// positive number before the decimal. At most three decimal places of precision
// are printed.
func (b bits) String() string {
	if b == 0 {
		return "0"
	}
	// Largest value is "-123.150EB"
	var buf [10]byte
	w := len(buf) - 1
	u := uint64(b)
	neg := b < 0
	if neg {
		u = -u
	}
	if u < uint64(byte_) {
		w -= 2
		copy(buf[w:], "bit")
		w = fmtInt(buf[:w], u)
	} else {
		switch {
		case u < uint64(kilobyte):
			w -= 0
			buf[w] = 'B'
			u = (u * 1e3 / 8)
		case u < uint64(megabyte):
			w -= 1
			copy(buf[w:], "kB")
			u /= 8
		case u < uint64(gigabyte):
			w -= 1
			copy(buf[w:], "MB")
			u /= 8 * 1e3
		case u < uint64(terabyte):
			w -= 1
			copy(buf[w:], "GB")
			u /= 8 * 1e6
		case u < uint64(petabyte):
			w -= 1
			copy(buf[w:], "TB")
			u /= 8 * 1e9
		case u < uint64(exabyte):
			w -= 1
			copy(buf[w:], "PB")
			u /= 8 * 1e12
		case u >= uint64(exabyte):
			w -= 1
			copy(buf[w:], "EB")
			u /= 8 * 1e15
		}
		w, u = fmtFrac(buf[:w], u, 3)
		w = fmtInt(buf[:w], u)
	}
	if neg {
		w--
		buf[w] = '-'
	}
	return string(buf[w:])
}
