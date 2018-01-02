package influxql

import (
	"bytes"
	"regexp"
)

var (
	sanitizeSetPassword = regexp.MustCompile(`(?i)password\s+for[^=]*=\s+(["']?[^\s"]+["']?)`)

	sanitizeCreatePassword = regexp.MustCompile(`(?i)with\s+password\s+(["']?[^\s"]+["']?)`)
)

// Sanitize attempts to sanitize passwords out of a raw query.
// It looks for patterns that may be related to the SET PASSWORD and CREATE USER
// statements and will redact the password that should be there. It will attempt
// to redact information from common invalid queries too, but it's not guaranteed
// to succeed on improper queries.
//
// This function works on the raw query and attempts to retain the original input
// as much as possible.
func Sanitize(query string) string {
	if matches := sanitizeSetPassword.FindAllStringSubmatchIndex(query, -1); matches != nil {
		var buf bytes.Buffer
		i := 0
		for _, match := range matches {
			buf.WriteString(query[i:match[2]])
			buf.WriteString("[REDACTED]")
			i = match[3]
		}
		buf.WriteString(query[i:])
		query = buf.String()
	}

	if matches := sanitizeCreatePassword.FindAllStringSubmatchIndex(query, -1); matches != nil {
		var buf bytes.Buffer
		i := 0
		for _, match := range matches {
			buf.WriteString(query[i:match[2]])
			buf.WriteString("[REDACTED]")
			i = match[3]
		}
		buf.WriteString(query[i:])
		query = buf.String()
	}
	return query
}
