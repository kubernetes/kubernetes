package aec

import (
	"fmt"
)

// RGB3Bit is a 3bit RGB color.
type RGB3Bit uint8

// RGB8Bit is a 8bit RGB color.
type RGB8Bit uint8

func newSGR(n uint) ANSI {
	return newAnsi(fmt.Sprintf(esc+"%dm", n))
}

// NewRGB3Bit create a RGB3Bit from given RGB.
func NewRGB3Bit(r, g, b uint8) RGB3Bit {
	return RGB3Bit((r >> 7) | ((g >> 6) & 0x2) | ((b >> 5) & 0x4))
}

// NewRGB8Bit create a RGB8Bit from given RGB.
func NewRGB8Bit(r, g, b uint8) RGB8Bit {
	return RGB8Bit(16 + 36*(r/43) + 6*(g/43) + b/43)
}

// Color3BitF set the foreground color of text.
func Color3BitF(c RGB3Bit) ANSI {
	return newAnsi(fmt.Sprintf(esc+"%dm", c+30))
}

// Color3BitB set the background color of text.
func Color3BitB(c RGB3Bit) ANSI {
	return newAnsi(fmt.Sprintf(esc+"%dm", c+40))
}

// Color8BitF set the foreground color of text.
func Color8BitF(c RGB8Bit) ANSI {
	return newAnsi(fmt.Sprintf(esc+"38;5;%dm", c))
}

// Color8BitB set the background color of text.
func Color8BitB(c RGB8Bit) ANSI {
	return newAnsi(fmt.Sprintf(esc+"48;5;%dm", c))
}

// FullColorF set the foreground color of text.
func FullColorF(r, g, b uint8) ANSI {
	return newAnsi(fmt.Sprintf(esc+"38;2;%d;%d;%dm", r, g, b))
}

// FullColorB set the foreground color of text.
func FullColorB(r, g, b uint8) ANSI {
	return newAnsi(fmt.Sprintf(esc+"48;2;%d;%d;%dm", r, g, b))
}

// Style
var (
	// Bold set the text style to bold or increased intensity.
	Bold ANSI

	// Faint set the text style to faint.
	Faint ANSI

	// Italic set the text style to italic.
	Italic ANSI

	// Underline set the text style to underline.
	Underline ANSI

	// BlinkSlow set the text style to slow blink.
	BlinkSlow ANSI

	// BlinkRapid set the text style to rapid blink.
	BlinkRapid ANSI

	// Inverse swap the foreground color and background color.
	Inverse ANSI

	// Conceal set the text style to conceal.
	Conceal ANSI

	// CrossOut set the text style to crossed out.
	CrossOut ANSI

	// Frame set the text style to framed.
	Frame ANSI

	// Encircle set the text style to encircled.
	Encircle ANSI

	// Overline set the text style to overlined.
	Overline ANSI
)

// Foreground color of text.
var (
	// DefaultF is the default color of foreground.
	DefaultF ANSI

	// Normal color
	BlackF   ANSI
	RedF     ANSI
	GreenF   ANSI
	YellowF  ANSI
	BlueF    ANSI
	MagentaF ANSI
	CyanF    ANSI
	WhiteF   ANSI

	// Light color
	LightBlackF   ANSI
	LightRedF     ANSI
	LightGreenF   ANSI
	LightYellowF  ANSI
	LightBlueF    ANSI
	LightMagentaF ANSI
	LightCyanF    ANSI
	LightWhiteF   ANSI
)

// Background color of text.
var (
	// DefaultB is the default color of background.
	DefaultB ANSI

	// Normal color
	BlackB   ANSI
	RedB     ANSI
	GreenB   ANSI
	YellowB  ANSI
	BlueB    ANSI
	MagentaB ANSI
	CyanB    ANSI
	WhiteB   ANSI

	// Light color
	LightBlackB   ANSI
	LightRedB     ANSI
	LightGreenB   ANSI
	LightYellowB  ANSI
	LightBlueB    ANSI
	LightMagentaB ANSI
	LightCyanB    ANSI
	LightWhiteB   ANSI
)

func init() {
	Bold = newSGR(1)
	Faint = newSGR(2)
	Italic = newSGR(3)
	Underline = newSGR(4)
	BlinkSlow = newSGR(5)
	BlinkRapid = newSGR(6)
	Inverse = newSGR(7)
	Conceal = newSGR(8)
	CrossOut = newSGR(9)

	BlackF = newSGR(30)
	RedF = newSGR(31)
	GreenF = newSGR(32)
	YellowF = newSGR(33)
	BlueF = newSGR(34)
	MagentaF = newSGR(35)
	CyanF = newSGR(36)
	WhiteF = newSGR(37)

	DefaultF = newSGR(39)

	BlackB = newSGR(40)
	RedB = newSGR(41)
	GreenB = newSGR(42)
	YellowB = newSGR(43)
	BlueB = newSGR(44)
	MagentaB = newSGR(45)
	CyanB = newSGR(46)
	WhiteB = newSGR(47)

	DefaultB = newSGR(49)

	Frame = newSGR(51)
	Encircle = newSGR(52)
	Overline = newSGR(53)

	LightBlackF = newSGR(90)
	LightRedF = newSGR(91)
	LightGreenF = newSGR(92)
	LightYellowF = newSGR(93)
	LightBlueF = newSGR(94)
	LightMagentaF = newSGR(95)
	LightCyanF = newSGR(96)
	LightWhiteF = newSGR(97)

	LightBlackB = newSGR(100)
	LightRedB = newSGR(101)
	LightGreenB = newSGR(102)
	LightYellowB = newSGR(103)
	LightBlueB = newSGR(104)
	LightMagentaB = newSGR(105)
	LightCyanB = newSGR(106)
	LightWhiteB = newSGR(107)
}
