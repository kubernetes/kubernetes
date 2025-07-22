package humanize

import (
	"fmt"
	"math/big"
	"strings"
	"unicode"
)

var (
	bigIECExp = big.NewInt(1024)

	// BigByte is one byte in bit.Ints
	BigByte = big.NewInt(1)
	// BigKiByte is 1,024 bytes in bit.Ints
	BigKiByte = (&big.Int{}).Mul(BigByte, bigIECExp)
	// BigMiByte is 1,024 k bytes in bit.Ints
	BigMiByte = (&big.Int{}).Mul(BigKiByte, bigIECExp)
	// BigGiByte is 1,024 m bytes in bit.Ints
	BigGiByte = (&big.Int{}).Mul(BigMiByte, bigIECExp)
	// BigTiByte is 1,024 g bytes in bit.Ints
	BigTiByte = (&big.Int{}).Mul(BigGiByte, bigIECExp)
	// BigPiByte is 1,024 t bytes in bit.Ints
	BigPiByte = (&big.Int{}).Mul(BigTiByte, bigIECExp)
	// BigEiByte is 1,024 p bytes in bit.Ints
	BigEiByte = (&big.Int{}).Mul(BigPiByte, bigIECExp)
	// BigZiByte is 1,024 e bytes in bit.Ints
	BigZiByte = (&big.Int{}).Mul(BigEiByte, bigIECExp)
	// BigYiByte is 1,024 z bytes in bit.Ints
	BigYiByte = (&big.Int{}).Mul(BigZiByte, bigIECExp)
	// BigRiByte is 1,024 y bytes in bit.Ints
	BigRiByte = (&big.Int{}).Mul(BigYiByte, bigIECExp)
	// BigQiByte is 1,024 r bytes in bit.Ints
	BigQiByte = (&big.Int{}).Mul(BigRiByte, bigIECExp)
)

var (
	bigSIExp = big.NewInt(1000)

	// BigSIByte is one SI byte in big.Ints
	BigSIByte = big.NewInt(1)
	// BigKByte is 1,000 SI bytes in big.Ints
	BigKByte = (&big.Int{}).Mul(BigSIByte, bigSIExp)
	// BigMByte is 1,000 SI k bytes in big.Ints
	BigMByte = (&big.Int{}).Mul(BigKByte, bigSIExp)
	// BigGByte is 1,000 SI m bytes in big.Ints
	BigGByte = (&big.Int{}).Mul(BigMByte, bigSIExp)
	// BigTByte is 1,000 SI g bytes in big.Ints
	BigTByte = (&big.Int{}).Mul(BigGByte, bigSIExp)
	// BigPByte is 1,000 SI t bytes in big.Ints
	BigPByte = (&big.Int{}).Mul(BigTByte, bigSIExp)
	// BigEByte is 1,000 SI p bytes in big.Ints
	BigEByte = (&big.Int{}).Mul(BigPByte, bigSIExp)
	// BigZByte is 1,000 SI e bytes in big.Ints
	BigZByte = (&big.Int{}).Mul(BigEByte, bigSIExp)
	// BigYByte is 1,000 SI z bytes in big.Ints
	BigYByte = (&big.Int{}).Mul(BigZByte, bigSIExp)
	// BigRByte is 1,000 SI y bytes in big.Ints
	BigRByte = (&big.Int{}).Mul(BigYByte, bigSIExp)
	// BigQByte is 1,000 SI r bytes in big.Ints
	BigQByte = (&big.Int{}).Mul(BigRByte, bigSIExp)
)

var bigBytesSizeTable = map[string]*big.Int{
	"b":   BigByte,
	"kib": BigKiByte,
	"kb":  BigKByte,
	"mib": BigMiByte,
	"mb":  BigMByte,
	"gib": BigGiByte,
	"gb":  BigGByte,
	"tib": BigTiByte,
	"tb":  BigTByte,
	"pib": BigPiByte,
	"pb":  BigPByte,
	"eib": BigEiByte,
	"eb":  BigEByte,
	"zib": BigZiByte,
	"zb":  BigZByte,
	"yib": BigYiByte,
	"yb":  BigYByte,
	"rib": BigRiByte,
	"rb":  BigRByte,
	"qib": BigQiByte,
	"qb":  BigQByte,
	// Without suffix
	"":   BigByte,
	"ki": BigKiByte,
	"k":  BigKByte,
	"mi": BigMiByte,
	"m":  BigMByte,
	"gi": BigGiByte,
	"g":  BigGByte,
	"ti": BigTiByte,
	"t":  BigTByte,
	"pi": BigPiByte,
	"p":  BigPByte,
	"ei": BigEiByte,
	"e":  BigEByte,
	"z":  BigZByte,
	"zi": BigZiByte,
	"y":  BigYByte,
	"yi": BigYiByte,
	"r":  BigRByte,
	"ri": BigRiByte,
	"q":  BigQByte,
	"qi": BigQiByte,
}

var ten = big.NewInt(10)

func humanateBigBytes(s, base *big.Int, sizes []string) string {
	if s.Cmp(ten) < 0 {
		return fmt.Sprintf("%d B", s)
	}
	c := (&big.Int{}).Set(s)
	val, mag := oomm(c, base, len(sizes)-1)
	suffix := sizes[mag]
	f := "%.0f %s"
	if val < 10 {
		f = "%.1f %s"
	}

	return fmt.Sprintf(f, val, suffix)

}

// BigBytes produces a human readable representation of an SI size.
//
// See also: ParseBigBytes.
//
// BigBytes(82854982) -> 83 MB
func BigBytes(s *big.Int) string {
	sizes := []string{"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"}
	return humanateBigBytes(s, bigSIExp, sizes)
}

// BigIBytes produces a human readable representation of an IEC size.
//
// See also: ParseBigBytes.
//
// BigIBytes(82854982) -> 79 MiB
func BigIBytes(s *big.Int) string {
	sizes := []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB"}
	return humanateBigBytes(s, bigIECExp, sizes)
}

// ParseBigBytes parses a string representation of bytes into the number
// of bytes it represents.
//
// See also: BigBytes, BigIBytes.
//
// ParseBigBytes("42 MB") -> 42000000, nil
// ParseBigBytes("42 mib") -> 44040192, nil
func ParseBigBytes(s string) (*big.Int, error) {
	lastDigit := 0
	hasComma := false
	for _, r := range s {
		if !(unicode.IsDigit(r) || r == '.' || r == ',') {
			break
		}
		if r == ',' {
			hasComma = true
		}
		lastDigit++
	}

	num := s[:lastDigit]
	if hasComma {
		num = strings.Replace(num, ",", "", -1)
	}

	val := &big.Rat{}
	_, err := fmt.Sscanf(num, "%f", val)
	if err != nil {
		return nil, err
	}

	extra := strings.ToLower(strings.TrimSpace(s[lastDigit:]))
	if m, ok := bigBytesSizeTable[extra]; ok {
		mv := (&big.Rat{}).SetInt(m)
		val.Mul(val, mv)
		rv := &big.Int{}
		rv.Div(val.Num(), val.Denom())
		return rv, nil
	}

	return nil, fmt.Errorf("unhandled size name: %v", extra)
}
