package humanize

import (
	"bytes"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// Comma produces a string form of the given number in base 10 with
// commas after every three orders of magnitude.
//
// e.g. Comma(834142) -> 834,142
func Comma(v int64) string {
	sign := ""

	// Min int64 can't be negated to a usable value, so it has to be special cased.
	if v == math.MinInt64 {
		return "-9,223,372,036,854,775,808"
	}

	if v < 0 {
		sign = "-"
		v = 0 - v
	}

	parts := []string{"", "", "", "", "", "", ""}
	j := len(parts) - 1

	for v > 999 {
		parts[j] = strconv.FormatInt(v%1000, 10)
		switch len(parts[j]) {
		case 2:
			parts[j] = "0" + parts[j]
		case 1:
			parts[j] = "00" + parts[j]
		}
		v = v / 1000
		j--
	}
	parts[j] = strconv.Itoa(int(v))
	return sign + strings.Join(parts[j:], ",")
}

// Commaf produces a string form of the given number in base 10 with
// commas after every three orders of magnitude.
//
// e.g. Commaf(834142.32) -> 834,142.32
func Commaf(v float64) string {
	buf := &bytes.Buffer{}
	if v < 0 {
		buf.Write([]byte{'-'})
		v = 0 - v
	}

	comma := []byte{','}

	parts := strings.Split(strconv.FormatFloat(v, 'f', -1, 64), ".")
	pos := 0
	if len(parts[0])%3 != 0 {
		pos += len(parts[0]) % 3
		buf.WriteString(parts[0][:pos])
		buf.Write(comma)
	}
	for ; pos < len(parts[0]); pos += 3 {
		buf.WriteString(parts[0][pos : pos+3])
		buf.Write(comma)
	}
	buf.Truncate(buf.Len() - 1)

	if len(parts) > 1 {
		buf.Write([]byte{'.'})
		buf.WriteString(parts[1])
	}
	return buf.String()
}

// CommafWithDigits works like the Commaf but limits the resulting
// string to the given number of decimal places.
//
// e.g. CommafWithDigits(834142.32, 1) -> 834,142.3
func CommafWithDigits(f float64, decimals int) string {
	return stripTrailingDigits(Commaf(f), decimals)
}

// BigComma produces a string form of the given big.Int in base 10
// with commas after every three orders of magnitude.
func BigComma(b *big.Int) string {
	sign := ""
	if b.Sign() < 0 {
		sign = "-"
		b.Abs(b)
	}

	athousand := big.NewInt(1000)
	c := (&big.Int{}).Set(b)
	_, m := oom(c, athousand)
	parts := make([]string, m+1)
	j := len(parts) - 1

	mod := &big.Int{}
	for b.Cmp(athousand) >= 0 {
		b.DivMod(b, athousand, mod)
		parts[j] = strconv.FormatInt(mod.Int64(), 10)
		switch len(parts[j]) {
		case 2:
			parts[j] = "0" + parts[j]
		case 1:
			parts[j] = "00" + parts[j]
		}
		j--
	}
	parts[j] = strconv.Itoa(int(b.Int64()))
	return sign + strings.Join(parts[j:], ",")
}
