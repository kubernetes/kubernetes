package internal

import (
	"bytes"
	"fmt"
	"io"
	"strings"
)

// ErrorWithLog returns an error which includes logs from the kernel verifier.
//
// The default error output is a summary of the full log. The latter can be
// accessed via VerifierError.Log or by formatting the error, see Format.
//
// A set of heuristics is used to determine whether the log has been truncated.
func ErrorWithLog(err error, log []byte) *VerifierError {
	const whitespace = "\t\r\v\n "

	// Convert verifier log C string by truncating it on the first 0 byte
	// and trimming trailing whitespace before interpreting as a Go string.
	truncated := false
	if i := bytes.IndexByte(log, 0); i != -1 {
		if i == len(log)-1 && !bytes.HasSuffix(log[:i], []byte{'\n'}) {
			// The null byte is at the end of the buffer and it's not preceded
			// by a newline character. Most likely the buffer was too short.
			truncated = true
		}

		log = log[:i]
	} else if len(log) > 0 {
		// No null byte? Dodgy!
		truncated = true
	}

	log = bytes.Trim(log, whitespace)
	logLines := bytes.Split(log, []byte{'\n'})
	lines := make([]string, 0, len(logLines))
	for _, line := range logLines {
		// Don't remove leading white space on individual lines. We rely on it
		// when outputting logs.
		lines = append(lines, string(bytes.TrimRight(line, whitespace)))
	}

	return &VerifierError{err, lines, truncated}
}

// VerifierError includes information from the eBPF verifier.
//
// It summarises the log output, see Format if you want to output the full contents.
type VerifierError struct {
	// The error which caused this error.
	Cause error
	// The verifier output split into lines.
	Log []string
	// Whether the log output is truncated, based on several heuristics.
	Truncated bool
}

func (le *VerifierError) Unwrap() error {
	return le.Cause
}

func (le *VerifierError) Error() string {
	log := le.Log
	if n := len(log); n > 0 && strings.HasPrefix(log[n-1], "processed ") {
		// Get rid of "processed 39 insns (limit 1000000) ..." from summary.
		log = log[:n-1]
	}

	n := len(log)
	if n == 0 {
		return le.Cause.Error()
	}

	lines := log[n-1:]
	if n >= 2 && (includePreviousLine(log[n-1]) || le.Truncated) {
		// Add one more line of context if it aids understanding the error.
		lines = log[n-2:]
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s: ", le.Cause.Error())

	for i, line := range lines {
		b.WriteString(strings.TrimSpace(line))
		if i != len(lines)-1 {
			b.WriteString(": ")
		}
	}

	omitted := len(le.Log) - len(lines)
	if omitted == 0 && !le.Truncated {
		return b.String()
	}

	b.WriteString(" (")
	if le.Truncated {
		b.WriteString("truncated")
	}

	if omitted > 0 {
		if le.Truncated {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d line(s) omitted", omitted)
	}
	b.WriteString(")")

	return b.String()
}

// includePreviousLine returns true if the given line likely is better
// understood with additional context from the preceding line.
func includePreviousLine(line string) bool {
	// We need to find a good trade off between understandable error messages
	// and too much complexity here. Checking the string prefix is ok, requiring
	// regular expressions to do it is probably overkill.

	if strings.HasPrefix(line, "\t") {
		// [13] STRUCT drm_rect size=16 vlen=4
		// \tx1 type_id=2
		return true
	}

	if len(line) >= 2 && line[0] == 'R' && line[1] >= '0' && line[1] <= '9' {
		// 0: (95) exit
		// R0 !read_ok
		return true
	}

	if strings.HasPrefix(line, "invalid bpf_context access") {
		// 0: (79) r6 = *(u64 *)(r1 +0)
		// func '__x64_sys_recvfrom' arg0 type FWD is not a struct
		// invalid bpf_context access off=0 size=8
		return true
	}

	return false
}

// Format the error.
//
// Understood verbs are %s and %v, which are equivalent to calling Error(). %v
// allows outputting additional information using the following flags:
//
//     +   Output the first <width> lines, or all lines if no width is given.
//     -   Output the last <width> lines, or all lines if no width is given.
//
// Use width to specify how many lines to output. Use the '-' flag to output
// lines from the end of the log instead of the beginning.
func (le *VerifierError) Format(f fmt.State, verb rune) {
	switch verb {
	case 's':
		_, _ = io.WriteString(f, le.Error())

	case 'v':
		n, haveWidth := f.Width()
		if !haveWidth || n > len(le.Log) {
			n = len(le.Log)
		}

		if !f.Flag('+') && !f.Flag('-') {
			if haveWidth {
				_, _ = io.WriteString(f, "%!v(BADWIDTH)")
				return
			}

			_, _ = io.WriteString(f, le.Error())
			return
		}

		if f.Flag('+') && f.Flag('-') {
			_, _ = io.WriteString(f, "%!v(BADFLAG)")
			return
		}

		fmt.Fprintf(f, "%s:", le.Cause.Error())

		omitted := len(le.Log) - n
		lines := le.Log[:n]
		if f.Flag('-') {
			// Print last instead of first lines.
			lines = le.Log[len(le.Log)-n:]
			if omitted > 0 {
				fmt.Fprintf(f, "\n\t(%d line(s) omitted)", omitted)
			}
		}

		for _, line := range lines {
			fmt.Fprintf(f, "\n\t%s", line)
		}

		if !f.Flag('-') {
			if omitted > 0 {
				fmt.Fprintf(f, "\n\t(%d line(s) omitted)", omitted)
			}
		}

		if le.Truncated {
			fmt.Fprintf(f, "\n\t(truncated)")
		}

	default:
		fmt.Fprintf(f, "%%!%c(BADVERB)", verb)
	}
}
