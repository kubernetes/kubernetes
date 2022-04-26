package runewidth

import (
	"os"
)

//go:generate go run script/generate.go

var (
	// EastAsianWidth will be set true if the current locale is CJK
	EastAsianWidth bool

	// ZeroWidthJoiner is flag to set to use UTR#51 ZWJ
	ZeroWidthJoiner bool

	// DefaultCondition is a condition in current locale
	DefaultCondition = &Condition{}
)

func init() {
	handleEnv()
}

func handleEnv() {
	env := os.Getenv("RUNEWIDTH_EASTASIAN")
	if env == "" {
		EastAsianWidth = IsEastAsian()
	} else {
		EastAsianWidth = env == "1"
	}
	// update DefaultCondition
	DefaultCondition.EastAsianWidth = EastAsianWidth
	DefaultCondition.ZeroWidthJoiner = ZeroWidthJoiner
}

type interval struct {
	first rune
	last  rune
}

type table []interval

func inTables(r rune, ts ...table) bool {
	for _, t := range ts {
		if inTable(r, t) {
			return true
		}
	}
	return false
}

func inTable(r rune, t table) bool {
	if r < t[0].first {
		return false
	}

	bot := 0
	top := len(t) - 1
	for top >= bot {
		mid := (bot + top) >> 1

		switch {
		case t[mid].last < r:
			bot = mid + 1
		case t[mid].first > r:
			top = mid - 1
		default:
			return true
		}
	}

	return false
}

var private = table{
	{0x00E000, 0x00F8FF}, {0x0F0000, 0x0FFFFD}, {0x100000, 0x10FFFD},
}

var nonprint = table{
	{0x0000, 0x001F}, {0x007F, 0x009F}, {0x00AD, 0x00AD},
	{0x070F, 0x070F}, {0x180B, 0x180E}, {0x200B, 0x200F},
	{0x2028, 0x202E}, {0x206A, 0x206F}, {0xD800, 0xDFFF},
	{0xFEFF, 0xFEFF}, {0xFFF9, 0xFFFB}, {0xFFFE, 0xFFFF},
}

// Condition have flag EastAsianWidth whether the current locale is CJK or not.
type Condition struct {
	EastAsianWidth  bool
	ZeroWidthJoiner bool
}

// NewCondition return new instance of Condition which is current locale.
func NewCondition() *Condition {
	return &Condition{
		EastAsianWidth:  EastAsianWidth,
		ZeroWidthJoiner: ZeroWidthJoiner,
	}
}

// RuneWidth returns the number of cells in r.
// See http://www.unicode.org/reports/tr11/
func (c *Condition) RuneWidth(r rune) int {
	switch {
	case r < 0 || r > 0x10FFFF || inTables(r, nonprint, combining, notassigned):
		return 0
	case (c.EastAsianWidth && IsAmbiguousWidth(r)) || inTables(r, doublewidth):
		return 2
	default:
		return 1
	}
}

func (c *Condition) stringWidth(s string) (width int) {
	for _, r := range []rune(s) {
		width += c.RuneWidth(r)
	}
	return width
}

func (c *Condition) stringWidthZeroJoiner(s string) (width int) {
	r1, r2 := rune(0), rune(0)
	for _, r := range []rune(s) {
		if r == 0xFE0E || r == 0xFE0F {
			continue
		}
		w := c.RuneWidth(r)
		if r2 == 0x200D && inTables(r, emoji) && inTables(r1, emoji) {
			if width < w {
				width = w
			}
		} else {
			width += w
		}
		r1, r2 = r2, r
	}
	return width
}

// StringWidth return width as you can see
func (c *Condition) StringWidth(s string) (width int) {
	if c.ZeroWidthJoiner {
		return c.stringWidthZeroJoiner(s)
	}
	return c.stringWidth(s)
}

// Truncate return string truncated with w cells
func (c *Condition) Truncate(s string, w int, tail string) string {
	if c.StringWidth(s) <= w {
		return s
	}
	r := []rune(s)
	tw := c.StringWidth(tail)
	w -= tw
	width := 0
	i := 0
	for ; i < len(r); i++ {
		cw := c.RuneWidth(r[i])
		if width+cw > w {
			break
		}
		width += cw
	}
	return string(r[0:i]) + tail
}

// Wrap return string wrapped with w cells
func (c *Condition) Wrap(s string, w int) string {
	width := 0
	out := ""
	for _, r := range []rune(s) {
		cw := RuneWidth(r)
		if r == '\n' {
			out += string(r)
			width = 0
			continue
		} else if width+cw > w {
			out += "\n"
			width = 0
			out += string(r)
			width += cw
			continue
		}
		out += string(r)
		width += cw
	}
	return out
}

// FillLeft return string filled in left by spaces in w cells
func (c *Condition) FillLeft(s string, w int) string {
	width := c.StringWidth(s)
	count := w - width
	if count > 0 {
		b := make([]byte, count)
		for i := range b {
			b[i] = ' '
		}
		return string(b) + s
	}
	return s
}

// FillRight return string filled in left by spaces in w cells
func (c *Condition) FillRight(s string, w int) string {
	width := c.StringWidth(s)
	count := w - width
	if count > 0 {
		b := make([]byte, count)
		for i := range b {
			b[i] = ' '
		}
		return s + string(b)
	}
	return s
}

// RuneWidth returns the number of cells in r.
// See http://www.unicode.org/reports/tr11/
func RuneWidth(r rune) int {
	return DefaultCondition.RuneWidth(r)
}

// IsAmbiguousWidth returns whether is ambiguous width or not.
func IsAmbiguousWidth(r rune) bool {
	return inTables(r, private, ambiguous)
}

// IsNeutralWidth returns whether is neutral width or not.
func IsNeutralWidth(r rune) bool {
	return inTable(r, neutral)
}

// StringWidth return width as you can see
func StringWidth(s string) (width int) {
	return DefaultCondition.StringWidth(s)
}

// Truncate return string truncated with w cells
func Truncate(s string, w int, tail string) string {
	return DefaultCondition.Truncate(s, w, tail)
}

// Wrap return string wrapped with w cells
func Wrap(s string, w int) string {
	return DefaultCondition.Wrap(s, w)
}

// FillLeft return string filled in left by spaces in w cells
func FillLeft(s string, w int) string {
	return DefaultCondition.FillLeft(s, w)
}

// FillRight return string filled in left by spaces in w cells
func FillRight(s string, w int) string {
	return DefaultCondition.FillRight(s, w)
}
