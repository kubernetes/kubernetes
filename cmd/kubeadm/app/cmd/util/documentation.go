/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"
)

var (
	// AlphaDisclaimer to be places at the end of description of commands in alpha release
	AlphaDisclaimer = `
		Alpha Disclaimer: this command is currently alpha.
	`

	// MacroCommandLongDescription provide a standard description for "macro" commands
	MacroCommandLongDescription = LongDesc(`
		This command is not meant to be run on its own. See list of available subcommands.
	`)
)

// LongDesc is designed to help with producing better long command line descriptions in code.
// Its behavior is somewhat inspired by the same function of kubectl, which uses Markdown for the input message.
// This one is not Markdown compliant, but it covers the needs of kubeadm. In particular:
// - Beginning and trailing space characters (including empty lines), are stripped from the output.
// - Consecutive non-empty lines of text are joined with spaces to form paragraphs.
// - Paragraphs are blocks of text divided by one or more empty lines or lines consisting only of "space" characters.
// - Paragraphs are spaced by precisely one empty line in the output.
// - A line break can be forced by adding a couple of empty spaces at the end of a text line.
// - All indentation is removed. The resulting output is not indented.
func LongDesc(s string) string {
	// Strip beginning and trailing space characters (including empty lines) and split the lines into a slice
	lines := strings.Split(strings.TrimSpace(s), "\n")

	output := []string{}
	paragraph := []string{}

	for _, line := range lines {
		// Remove indentation and trailing spaces from the current line
		trimmedLine := strings.TrimSpace(line)
		if trimmedLine == "" {
			if len(paragraph) > 0 {
				// If the line is empty and the current paragraph is not, we have reached a paragraph end.
				// (if the paragraph and the line are empty, then this is non-first empty line in between paragraphs and needs to be ignored)
				// In that case we join all of the paragraph lines with a single space,
				// add a trailing newline character (to ensure an empty line after the paragraph),
				// append the paragraph text to the output and clear everything in the current paragraph slice.
				output = append(output, strings.Join(paragraph, " ")+"\n")
				paragraph = []string{}
			}
		} else {
			// Non-empty text line, append it to the current paragraph
			paragraph = append(paragraph, trimmedLine)
			if strings.HasSuffix(line, "  ") {
				// If the original line has a suffix of couple of spaces, then we add a line break.
				// This is achieved by flushing out the current paragraph and starting a new one.
				// No trailing space is added to the flushed paragraph,
				// so that no empty line is placed between the old and the new paragraphs (a simple line break)
				output = append(output, strings.Join(paragraph, " "))
				paragraph = []string{}
			}
		}
	}

	// The last paragraph is always unflushed, so flush it.
	// We don't add a trailing newline character, so that we won't have to strip the output.
	output = append(output, strings.Join(paragraph, " "))

	// Join all the paragraphs together with new lines in between them.
	return strings.Join(output, "\n")
}

// Examples is designed to help with producing examples for command line usage.
// Its behavior is mimicking a similar kubectl function in the following ways:
// - Beginning and trailing space characters (including empty lines), are stripped from the output.
// - All lines of text are stripped of beginning and trailing spaces (thus loosing indentation) and are then double-space indented.
func Examples(s string) string {
	trimmedText := strings.TrimSpace(s)
	if trimmedText == "" {
		return ""
	}

	const indent = `  `
	inLines := strings.Split(trimmedText, "\n")
	outLines := make([]string, 0, len(inLines))

	for _, line := range inLines {
		outLines = append(outLines, indent+strings.TrimSpace(line))
	}

	return strings.Join(outLines, "\n")
}
