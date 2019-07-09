package colors

import (
	"fmt"
	"strings"
)

const ansiEscape = "\x1b"

// a color code type
type color int

// some ansi colors
const (
	black color = iota + 30
	red
	green
	yellow
	blue    // unused
	magenta // unused
	cyan
	white
)

func colorize(s interface{}, c color) string {
	return fmt.Sprintf("%s[%dm%v%s[0m", ansiEscape, c, s, ansiEscape)
}

type ColorFunc func(interface{}) string

func Bold(fn ColorFunc) ColorFunc {
	return ColorFunc(func(input interface{}) string {
		return strings.Replace(fn(input), ansiEscape+"[", ansiEscape+"[1;", 1)
	})
}

func Green(s interface{}) string {
	return colorize(s, green)
}

func Red(s interface{}) string {
	return colorize(s, red)
}

func Cyan(s interface{}) string {
	return colorize(s, cyan)
}

func Black(s interface{}) string {
	return colorize(s, black)
}

func Yellow(s interface{}) string {
	return colorize(s, yellow)
}

func White(s interface{}) string {
	return colorize(s, white)
}
