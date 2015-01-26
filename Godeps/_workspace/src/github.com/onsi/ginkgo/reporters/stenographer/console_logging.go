package stenographer

import (
	"fmt"
	"strings"
)

func (s *consoleStenographer) colorize(colorCode string, format string, args ...interface{}) string {
	var out string

	if len(args) > 0 {
		out = fmt.Sprintf(format, args...)
	} else {
		out = format
	}

	if s.color {
		return fmt.Sprintf("%s%s%s", colorCode, out, defaultStyle)
	} else {
		return out
	}
}

func (s *consoleStenographer) printBanner(text string, bannerCharacter string) {
	fmt.Println(text)
	fmt.Println(strings.Repeat(bannerCharacter, len(text)))
}

func (s *consoleStenographer) printNewLine() {
	fmt.Println("")
}

func (s *consoleStenographer) printDelimiter() {
	fmt.Println(s.colorize(grayColor, "%s", strings.Repeat("-", 30)))
}

func (s *consoleStenographer) print(indentation int, format string, args ...interface{}) {
	fmt.Print(s.indent(indentation, format, args...))
}

func (s *consoleStenographer) println(indentation int, format string, args ...interface{}) {
	fmt.Println(s.indent(indentation, format, args...))
}

func (s *consoleStenographer) indent(indentation int, format string, args ...interface{}) string {
	var text string

	if len(args) > 0 {
		text = fmt.Sprintf(format, args...)
	} else {
		text = format
	}

	stringArray := strings.Split(text, "\n")
	padding := ""
	if indentation >= 0 {
		padding = strings.Repeat("  ", indentation)
	}
	for i, s := range stringArray {
		stringArray[i] = fmt.Sprintf("%s%s", padding, s)
	}

	return strings.Join(stringArray, "\n")
}
