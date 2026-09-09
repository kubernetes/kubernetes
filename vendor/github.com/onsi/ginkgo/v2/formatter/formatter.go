package formatter

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// ColorableStdOut and ColorableStdErr enable color output support on Windows
var ColorableStdOut = newColorable(os.Stdout)
var ColorableStdErr = newColorable(os.Stderr)

const COLS = 80

type ColorMode uint8

const (
	ColorModeNone ColorMode = iota
	ColorModeTerminal
	ColorModePassthrough
)

var SingletonFormatter = New(ColorModeTerminal)

func F(format string, args ...any) string {
	return SingletonFormatter.F(format, args...)
}

func Fi(indentation uint, format string, args ...any) string {
	return SingletonFormatter.Fi(indentation, format, args...)
}

func Fiw(indentation uint, maxWidth uint, format string, args ...any) string {
	return SingletonFormatter.Fiw(indentation, maxWidth, format, args...)
}

type Formatter struct {
	ColorMode                ColorMode
	colors                   map[string]string
	styleRe                  *regexp.Regexp
	preserveColorStylingTags bool
}

func NewWithNoColorBool(noColor bool) Formatter {
	if noColor {
		return New(ColorModeNone)
	}
	return New(ColorModeTerminal)
}

func New(colorMode ColorMode) Formatter {
	colorAliases := map[string]int{
		"black":   0,
		"red":     1,
		"green":   2,
		"yellow":  3,
		"blue":    4,
		"magenta": 5,
		"cyan":    6,
		"white":   7,
	}
	for colorAlias, n := range colorAliases {
		colorAliases[fmt.Sprintf("bright-%s", colorAlias)] = n + 8
	}

	getColor := func(color, defaultEscapeCode string) string {
		color = strings.ToUpper(strings.ReplaceAll(color, "-", "_"))
		envVar := fmt.Sprintf("GINKGO_CLI_COLOR_%s", color)
		envVarColor := os.Getenv(envVar)
		if envVarColor == "" {
			return defaultEscapeCode
		}
		if colorCode, ok := colorAliases[envVarColor]; ok {
			return fmt.Sprintf("\x1b[38;5;%dm", colorCode)
		}
		colorCode, err := strconv.Atoi(envVarColor)
		if err != nil || colorCode < 0 || colorCode > 255 {
			return defaultEscapeCode
		}
		return fmt.Sprintf("\x1b[38;5;%dm", colorCode)
	}

	if _, noColor := os.LookupEnv("GINKGO_NO_COLOR"); noColor {
		colorMode = ColorModeNone
	}

	f := Formatter{
		ColorMode: colorMode,
		colors: map[string]string{
			"/":         "\x1b[0m",
			"bold":      "\x1b[1m",
			"underline": "\x1b[4m",

			"red":          getColor("red", "\x1b[38;5;9m"),
			"orange":       getColor("orange", "\x1b[38;5;214m"),
			"coral":        getColor("coral", "\x1b[38;5;204m"),
			"magenta":      getColor("magenta", "\x1b[38;5;13m"),
			"green":        getColor("green", "\x1b[38;5;10m"),
			"dark-green":   getColor("dark-green", "\x1b[38;5;28m"),
			"yellow":       getColor("yellow", "\x1b[38;5;11m"),
			"light-yellow": getColor("light-yellow", "\x1b[38;5;228m"),
			"cyan":         getColor("cyan", "\x1b[38;5;14m"),
			"gray":         getColor("gray", "\x1b[38;5;243m"),
			"light-gray":   getColor("light-gray", "\x1b[38;5;246m"),
			"blue":         getColor("blue", "\x1b[38;5;12m"),
		},
	}
	colors := []string{}
	for color := range f.colors {
		colors = append(colors, color)
	}
	f.styleRe = regexp.MustCompile("{{(" + strings.Join(colors, "|") + ")}}")
	return f
}

func (f Formatter) F(format string, args ...any) string {
	return f.Fi(0, format, args...)
}

func (f Formatter) Fi(indentation uint, format string, args ...any) string {
	return f.Fiw(indentation, 0, format, args...)
}

func (f Formatter) Fiw(indentation uint, maxWidth uint, format string, args ...any) string {
	out := f.style(format)
	if len(args) > 0 {
		out = fmt.Sprintf(out, args...)
	}

	if indentation == 0 && maxWidth == 0 {
		return out
	}

	lines := strings.Split(out, "\n")

	if maxWidth != 0 {
		outLines := []string{}

		maxWidth = maxWidth - indentation*2
		for _, line := range lines {
			if f.length(line) <= maxWidth {
				outLines = append(outLines, line)
				continue
			}
			words := strings.Split(line, " ")
			outWords := []string{words[0]}
			length := uint(f.length(words[0]))
			for _, word := range words[1:] {
				wordLength := f.length(word)
				if length+wordLength+1 <= maxWidth {
					length += wordLength + 1
					outWords = append(outWords, word)
					continue
				}
				outLines = append(outLines, strings.Join(outWords, " "))
				outWords = []string{word}
				length = wordLength
			}
			if len(outWords) > 0 {
				outLines = append(outLines, strings.Join(outWords, " "))
			}
		}

		lines = outLines
	}

	if indentation == 0 {
		return strings.Join(lines, "\n")
	}

	padding := strings.Repeat("  ", int(indentation))
	for i := range lines {
		if lines[i] != "" {
			lines[i] = padding + lines[i]
		}
	}

	return strings.Join(lines, "\n")
}

func (f Formatter) length(styled string) uint {
	n := uint(0)
	inStyle := false
	for _, b := range styled {
		if inStyle {
			if b == 'm' {
				inStyle = false
			}
			continue
		}
		if b == '\x1b' {
			inStyle = true
			continue
		}
		n += 1
	}
	return n
}

func (f Formatter) CycleJoin(elements []string, joiner string, cycle []string) string {
	if len(elements) == 0 {
		return ""
	}
	n := len(cycle)
	out := ""
	for i, text := range elements {
		out += cycle[i%n] + text
		if i < len(elements)-1 {
			out += joiner
		}
	}
	out += "{{/}}"
	return f.style(out)
}

func (f Formatter) style(s string) string {
	switch f.ColorMode {
	case ColorModeNone:
		return f.styleRe.ReplaceAllString(s, "")
	case ColorModePassthrough:
		return s
	case ColorModeTerminal:
		return f.styleRe.ReplaceAllStringFunc(s, func(match string) string {
			if out, ok := f.colors[strings.Trim(match, "{}")]; ok {
				return out
			}
			return match
		})
	}

	return ""
}
