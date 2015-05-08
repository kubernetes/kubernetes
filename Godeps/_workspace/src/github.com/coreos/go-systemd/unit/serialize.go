package unit

import (
	"bytes"
	"io"
)

// Serialize encodes all of the given UnitOption objects into a unit file
func Serialize(opts []*UnitOption) io.Reader {
	var buf bytes.Buffer

	if len(opts) == 0 {
		return &buf
	}

	curSection := opts[0].Section

	writeSectionHeader(&buf, curSection)
	writeNewline(&buf)

	for _, opt := range opts {
		if opt.Section != curSection {
			curSection = opt.Section

			writeNewline(&buf)
			writeSectionHeader(&buf, curSection)
			writeNewline(&buf)
		}

		writeOption(&buf, opt)
		writeNewline(&buf)
	}

	return &buf
}

func writeNewline(buf *bytes.Buffer) {
	buf.WriteRune('\n')
}

func writeSectionHeader(buf *bytes.Buffer, section string) {
	buf.WriteRune('[')
	buf.WriteString(section)
	buf.WriteRune(']')
}

func writeOption(buf *bytes.Buffer, opt *UnitOption) {
	buf.WriteString(opt.Name)
	buf.WriteRune('=')
	buf.WriteString(opt.Value)
}
