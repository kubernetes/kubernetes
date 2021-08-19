/*
Copyright 2021 The Kubernetes Authors.

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

package templates

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	wordwrap "github.com/mitchellh/go-wordwrap"
	flag "github.com/spf13/pflag"
)

const (
	tabwriterMinWidth = 0
	tabwriterTabWidth = 0
	tabwriterPadding  = 2
	tabwriterPadChar  = ' '
	tabwriterFlags    = 0
)

type HelpFlagPrinter struct {
	tabWriter *tabwriter.Writer
	out       io.Writer
}

// NewHelpFlagPrinter initializes golang's i/o writer and a tabwriter
func NewHelpFlagPrinter(out io.Writer) *HelpFlagPrinter {
	newFlagPrinter := &HelpFlagPrinter{out: out}
	newFlagPrinter.tabWriter = tabwriter.NewWriter(
		newFlagPrinter.out,
		tabwriterMinWidth,
		tabwriterTabWidth,
		tabwriterPadding,
		tabwriterPadChar,
		tabwriterFlags)
	return newFlagPrinter
}

// PrintHelpFlags will format the help flags using wordwrapper and tabwriter.
// The idea is to wrap the usage messages into the same width even if it's split into multiple lines.
// This can be done by calculating maximum characters that will be padded by tabwriter and take the difference with wrapLimit.
// Wrapping the usage messages into the same width will result in long help flags having short usage messages,
// therefore, we will pad it with extraChar to allow more characters for longer flags
// Note: It will only format the flags if the wrapLimit is greater or equal to 100
func (printer *HelpFlagPrinter) PrintHelpFlags(flag *flag.Flag, wrapLimit, tabWidth, minFlagLen, maxFlagLen uint) {
	formatBuf := new(bytes.Buffer)
	if wrapLimit >= 100 {
		fmt.Fprintf(formatBuf, getFlagFormatWithTab(flag), flag.Shorthand, flag.Name, flag.DefValue, flag.Usage)

		curFlagLen := uint(len(formatBuf.String()) - len(flag.Usage))
		extraChar := uint(curFlagLen - minFlagLen - 5)
		maxWidthAllowed := wrapLimit - tabWidth

		wrappedStr := wordwrap.WrapString(formatBuf.String(), maxWidthAllowed+extraChar)
		splitUsage := strings.Split(wrappedStr, "\n")
		wrappedFirstLine := splitUsage[0]
		// if the flag usage is longer than one line, wrap it again
		if len(splitUsage) > 1 {
			nextLinesSplit := splitUsage[1:]
			nextLines := strings.Join(nextLinesSplit, " ")
			// when wrapping again, unlike first line, we will ignore the current flag string length
			wrappedNextLines := wordwrap.WrapString(nextLines, maxWidthAllowed-curFlagLen+extraChar)
			wrappedStr = wrappedFirstLine + "\n" + wrappedNextLines
		}
		// append all next line characters with tab for the tabwriter
		appendTabStr := strings.ReplaceAll(wrappedStr, "\n", "\n\t")

		fmt.Fprintf(printer.tabWriter, appendTabStr+"\n")
	} else {
		fmt.Fprintf(formatBuf, getFlagFormatWithoutTab(flag), flag.Shorthand, flag.Name, flag.DefValue, flag.Usage)
		wrappedStr := wordwrap.WrapString(formatBuf.String(), wrapLimit)

		fmt.Fprintf(printer.out, wrappedStr+"\n")
	}
}

//FlushTabWriter will flush the tabWriter's buffer to HelpFlagPrinter's out
func (printer *HelpFlagPrinter) FlushTabWriter() {
	printer.tabWriter.Flush()
}
