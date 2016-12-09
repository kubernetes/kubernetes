/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"

	log "github.com/Sirupsen/logrus"
)

// Attribute represents ansi codes used for formatting, coloring, etc.
type Color int

const (
	// Base code used to disable colored output
	nocolor Color = 0

	// Escape codes used for coloring output
	escape                = "\x1b"
	extendedPalettePrefix = "38;5"

	// Color table available at: http://misc.flogisoft.com/_media/bash/colors_format/256_colors_fg.png
	black  Color = 0
	red    Color = 196
	orange Color = 208
	blue   Color = 39
	green  Color = 34
)

var supportsColorPalette = SupportsColorPalette()

// Implements logrus TextFormatter
type LogFormatter struct {
	// Set to true to disable timestamp logging (useful when the output
	// is redirected to a logging system already adding a timestamp)
	DisableTimestamp bool
}

func (l LogFormatter) Format(entry *log.Entry) ([]byte, error) {
	var keys []string
	for k := range entry.Data {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	b := &bytes.Buffer{}
	timestamp := entry.Time.Format("15:04:05")
	caller := getCaller()

	if supportsColorPalette {
		timestamp = colorizeString(green, timestamp)
		caller = colorizeString(blue, caller)
		entry.Message = colorizeString(getLevelColor(entry.Level), entry.Message)
	}

	if !l.DisableTimestamp {
		fmt.Fprintf(b, "%v ", timestamp)
	}

	fmt.Fprintf(b, "%v ", caller)
	fmt.Fprintf(b, "%v ", entry.Message)
	for _, key := range keys {
		fmt.Fprintf(b, "%v ", entry.Data[key])
	}

	b.WriteByte('\n')
	return b.Bytes(), nil
}

func getLevelColor(level log.Level) Color {
	switch level {
	case log.WarnLevel:
		return orange
	case log.ErrorLevel:
		return red
	default:
		return nocolor
	}
}

// Adds given color to the provided string.
func colorizeString(color Color, str string) string {
	if color == nocolor {
		return str
	}

	enableColor := fmt.Sprintf("%s[%s;%sm", escape, extendedPalettePrefix,
		strconv.Itoa(int(color)))
	disableColor := fmt.Sprintf("%s[%dm", escape, nocolor)

	return fmt.Sprintf("%s%s%s", enableColor, str, disableColor)
}

// Returns original caller info in format: '[dir/file]'.
func getCaller() string {
	const callerDepth = 9

	if _, file, _, ok := runtime.Caller(callerDepth); ok && len(file) > 0 {
		// Remove file extension and split file path
		parts := strings.Split(strings.Replace(file, ".go", "", 1),
			string(os.PathSeparator))

		// Take only last 2 parts from the path [dir/file]
		return fmt.Sprint("[", strings.Join(parts[len(parts)-2:],
			string(os.PathSeparator)), "]")
	}

	return ""
}

func needsQuoting(text string) bool {
	for _, ch := range text {
		if !((ch >= 'a' && ch <= 'z') ||
			(ch >= 'A' && ch <= 'Z') ||
			(ch >= '0' && ch < '9') ||
			ch == '-' || ch == '.') {
			return false
		}
	}
	return true
}

func (f *LogFormatter) appendKeyValue(b *bytes.Buffer, key, value interface{}) {
	fmt.Fprintf(b, "%v ", value)
}

func SupportsColorPalette() bool {
	// minimum required color number supported by tty
	requiredColorPalette := 8

	if !log.IsTerminal() {
		return false
	}

	// Get info about supported color palette from terminfo db
	out, err := exec.Command("tput", "colors").Output()
	if err != nil {
		return false
	}

	ncolors, err := strconv.Atoi(strings.Trim(string(out), "\n"))
	if err != nil || ncolors < requiredColorPalette {
		return false
	}

	return true
}
