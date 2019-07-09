package godog

import (
	"strings"
	"time"

	"github.com/DATA-DOG/godog/colors"
)

var (
	red    = colors.Red
	redb   = colors.Bold(colors.Red)
	green  = colors.Green
	black  = colors.Black
	yellow = colors.Yellow
	cyan   = colors.Cyan
	cyanb  = colors.Bold(colors.Cyan)
	whiteb = colors.Bold(colors.White)
)

// repeats a space n times
func s(n int) string {
	return strings.Repeat(" ", n)
}

var timeNowFunc = func() time.Time {
	return time.Now()
}
