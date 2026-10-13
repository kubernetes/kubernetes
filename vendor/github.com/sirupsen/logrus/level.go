package logrus

import (
	"strings"
	"sync"
)

const (
	ansiReset    = "\x1b[0m"    // reset attributes
	ansiRed      = "\x1b[31m"   // red
	ansiYellow   = "\x1b[33m"   // yellow
	ansiCyan     = "\x1b[36m"   // cyan
	ansiDimCyan  = "\x1b[2;36m" // dim cyan
	ansiDimWhite = "\x1b[2;37m" // dim white (light gray)
)

type lvlPrefix struct {
	full      string
	truncated string
	padded    string
}

func colorize(level Level, s string) string {
	color := ansiCyan
	switch level {
	case TraceLevel:
		color = ansiDimWhite
	case DebugLevel:
		color = ansiDimCyan
	case WarnLevel:
		color = ansiYellow
	case ErrorLevel, FatalLevel, PanicLevel:
		color = ansiRed
	case InfoLevel:
		color = ansiCyan
	}
	return color + s + ansiReset
}

func formatLevel(level Level, disableTrunc, pad bool, maxLen int) string {
	upper := strings.ToUpper(level.String())

	if pad && maxLen > len(upper) {
		upper += strings.Repeat(" ", maxLen-len(upper))
	}

	if !pad && !disableTrunc && len(upper) > 4 {
		upper = upper[:4]
	}

	return colorize(level, upper)
}

var levelPrefixOnce = sync.OnceValues(func() (map[Level]lvlPrefix, lvlPrefix) {
	var maxLevel Level
	maxLen := 0
	for _, lvl := range AllLevels {
		if lvl > maxLevel {
			maxLevel = lvl
		}
		if l := len(lvl.String()); l > maxLen {
			maxLen = l
		}
	}

	prefix := make(map[Level]lvlPrefix, len(AllLevels))
	for _, lvl := range AllLevels {
		prefix[lvl] = lvlPrefix{
			full:      formatLevel(lvl, true, false, maxLen),
			truncated: formatLevel(lvl, false, false, maxLen),
			padded:    formatLevel(lvl, true, true, maxLen),
		}
	}

	unknownLevel := maxLevel + 1
	unknown := lvlPrefix{
		full:      formatLevel(unknownLevel, true, false, maxLen),
		truncated: formatLevel(unknownLevel, false, false, maxLen),
		padded:    formatLevel(unknownLevel, true, true, maxLen),
	}

	return prefix, unknown
})

func levelPrefix(level Level, disableTrunc, pad bool) string {
	prefix, unknown := levelPrefixOnce()

	p, ok := prefix[level]
	if !ok {
		p = unknown
	}

	switch {
	case pad:
		return p.padded
	case !disableTrunc:
		return p.truncated
	default:
		return p.full
	}
}
