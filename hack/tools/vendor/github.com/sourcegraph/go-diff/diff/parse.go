package diff

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ParseMultiFileDiff parses a multi-file unified diff. It returns an error if
// parsing failed as a whole, but does its best to parse as many files in the
// case of per-file errors. If it cannot detect when the diff of the next file
// begins, the hunks are added to the FileDiff of the previous file.
func ParseMultiFileDiff(diff []byte) ([]*FileDiff, error) {
	return NewMultiFileDiffReader(bytes.NewReader(diff)).ReadAllFiles()
}

// NewMultiFileDiffReader returns a new MultiFileDiffReader that reads
// a multi-file unified diff from r.
func NewMultiFileDiffReader(r io.Reader) *MultiFileDiffReader {
	return &MultiFileDiffReader{reader: bufio.NewReader(r)}
}

// MultiFileDiffReader reads a multi-file unified diff.
type MultiFileDiffReader struct {
	line   int
	offset int64
	reader *bufio.Reader

	// TODO(sqs): line and offset tracking in multi-file diffs is broken; add tests and fix

	// nextFileFirstLine is a line that was read by a HunksReader that
	// was how it determined the hunk was complete. But to determine
	// that, it needed to read the first line of the next file. We
	// store nextFileFirstLine so we can "give the first line back" to
	// the next file.
	nextFileFirstLine []byte
}

// ReadFile reads the next file unified diff (including headers and
// all hunks) from r. If there are no more files in the diff, it
// returns error io.EOF.
func (r *MultiFileDiffReader) ReadFile() (*FileDiff, error) {
	fr := &FileDiffReader{
		line:           r.line,
		offset:         r.offset,
		reader:         r.reader,
		fileHeaderLine: r.nextFileFirstLine,
	}
	r.nextFileFirstLine = nil

	fd, err := fr.ReadAllHeaders()
	if err != nil {
		switch e := err.(type) {
		case *ParseError:
			if e.Err == ErrNoFileHeader || e.Err == ErrExtendedHeadersEOF {
				return nil, io.EOF
			}
			return nil, err

		case OverflowError:
			r.nextFileFirstLine = []byte(e)
			return fd, nil

		default:
			return nil, err
		}
	}

	// FileDiff is added/deleted file
	// No further collection of hunks needed
	if fd.NewName == "" {
		return fd, nil
	}

	// Before reading hunks, check to see if there are any. If there
	// aren't any, and there's another file after this file in the
	// diff, then the hunks reader will complain ErrNoHunkHeader. It's
	// not easy for us to tell from that error alone if that was
	// caused by the lack of any hunks, or a malformatted hunk, so we
	// need to perform the check here.
	hr := fr.HunksReader()
	line, err := readLine(r.reader)
	if err != nil && err != io.EOF {
		return fd, err
	}
	line = bytes.TrimSuffix(line, []byte{'\n'})
	if bytes.HasPrefix(line, hunkPrefix) {
		hr.nextHunkHeaderLine = line
		fd.Hunks, err = hr.ReadAllHunks()
		r.line = fr.line
		r.offset = fr.offset
		if err != nil {
			if e0, ok := err.(*ParseError); ok {
				if e, ok := e0.Err.(*ErrBadHunkLine); ok {
					// This just means we finished reading the hunks for the
					// current file. See the ErrBadHunkLine doc for more info.
					r.nextFileFirstLine = e.Line
					return fd, nil
				}
			}
			return nil, err
		}
	} else {
		// There weren't any hunks, so that line we peeked ahead at
		// actually belongs to the next file. Put it back.
		r.nextFileFirstLine = line
	}

	return fd, nil
}

// ReadAllFiles reads all file unified diffs (including headers and all
// hunks) remaining in r.
func (r *MultiFileDiffReader) ReadAllFiles() ([]*FileDiff, error) {
	var ds []*FileDiff
	for {
		d, err := r.ReadFile()
		if d != nil {
			ds = append(ds, d)
		}
		if err == io.EOF {
			return ds, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

// ParseFileDiff parses a file unified diff.
func ParseFileDiff(diff []byte) (*FileDiff, error) {
	return NewFileDiffReader(bytes.NewReader(diff)).Read()
}

// NewFileDiffReader returns a new FileDiffReader that reads a file
// unified diff.
func NewFileDiffReader(r io.Reader) *FileDiffReader {
	return &FileDiffReader{reader: bufio.NewReader(r)}
}

// FileDiffReader reads a unified file diff.
type FileDiffReader struct {
	line   int
	offset int64
	reader *bufio.Reader

	// fileHeaderLine is the first file header line, set by:
	//
	// (1) ReadExtendedHeaders if it encroaches on a file header line
	//     (which it must to detect when extended headers are done); or
	// (2) (*MultiFileDiffReader).ReadFile() if it encroaches on a
	//     file header line while reading the previous file's hunks (in a
	//     multi-file diff).
	fileHeaderLine []byte
}

// Read reads a file unified diff, including headers and hunks, from r.
func (r *FileDiffReader) Read() (*FileDiff, error) {
	fd, err := r.ReadAllHeaders()
	if err != nil {
		return nil, err
	}

	fd.Hunks, err = r.HunksReader().ReadAllHunks()
	if err != nil {
		return nil, err
	}

	return fd, nil
}

// ReadAllHeaders reads the file headers and extended headers (if any)
// from a file unified diff. It does not read hunks, and the returned
// FileDiff's Hunks field is nil. To read the hunks, call the
// (*FileDiffReader).HunksReader() method to get a HunksReader and
// read hunks from that.
func (r *FileDiffReader) ReadAllHeaders() (*FileDiff, error) {
	var err error
	fd := &FileDiff{}

	fd.Extended, err = r.ReadExtendedHeaders()
	if pe, ok := err.(*ParseError); ok && pe.Err == ErrExtendedHeadersEOF {
		wasEmpty := handleEmpty(fd)
		if wasEmpty {
			return fd, nil
		}
		return fd, err
	} else if _, ok := err.(OverflowError); ok {
		handleEmpty(fd)
		return fd, err
	} else if err != nil {
		return fd, err
	}

	var origTime, newTime *time.Time
	fd.OrigName, fd.NewName, origTime, newTime, err = r.ReadFileHeaders()
	if err != nil {
		return nil, err
	}
	if origTime != nil {
		fd.OrigTime = origTime
	}
	if newTime != nil {
		fd.NewTime = newTime
	}

	return fd, nil
}

// HunksReader returns a new HunksReader that reads hunks from r. The
// HunksReader's line and offset (used in error messages) is set to
// start where the file diff header ended (which means errors have the
// correct position information).
func (r *FileDiffReader) HunksReader() *HunksReader {
	return &HunksReader{
		line:   r.line,
		offset: r.offset,
		reader: r.reader,
	}
}

// ReadFileHeaders reads the unified file diff header (the lines that
// start with "---" and "+++" with the orig/new file names and
// timestamps). Or which starts with "Only in " with dir path and filename.
// "Only in" message is supported in POSIX locale: https://pubs.opengroup.org/onlinepubs/9699919799/utilities/diff.html#tag_20_34_10
func (r *FileDiffReader) ReadFileHeaders() (origName, newName string, origTimestamp, newTimestamp *time.Time, err error) {
	if r.fileHeaderLine != nil {
		if isOnlyMessage, source, filename := parseOnlyInMessage(r.fileHeaderLine); isOnlyMessage {
			return filepath.Join(string(source), string(filename)),
				"", nil, nil, nil
		}
	}

	origName, origTimestamp, err = r.readOneFileHeader([]byte("--- "))
	if err != nil {
		return "", "", nil, nil, err
	}

	newName, newTimestamp, err = r.readOneFileHeader([]byte("+++ "))
	if err != nil {
		return "", "", nil, nil, err
	}

	unquotedOrigName, err := strconv.Unquote(origName)
	if err == nil {
		origName = unquotedOrigName
	}
	unquotedNewName, err := strconv.Unquote(newName)
	if err == nil {
		newName = unquotedNewName
	}

	return origName, newName, origTimestamp, newTimestamp, nil
}

// readOneFileHeader reads one of the file headers (prefix should be
// either "+++ " or "--- ").
func (r *FileDiffReader) readOneFileHeader(prefix []byte) (filename string, timestamp *time.Time, err error) {
	var line []byte

	if r.fileHeaderLine == nil {
		var err error
		line, err = readLine(r.reader)
		if err == io.EOF {
			return "", nil, &ParseError{r.line, r.offset, ErrNoFileHeader}
		} else if err != nil {
			return "", nil, err
		}
	} else {
		line = r.fileHeaderLine
		r.fileHeaderLine = nil
	}

	if !bytes.HasPrefix(line, prefix) {
		return "", nil, &ParseError{r.line, r.offset, ErrBadFileHeader}
	}

	r.offset += int64(len(line))
	r.line++
	line = line[len(prefix):]

	trimmedLine := strings.TrimSpace(string(line)) // filenames that contain spaces may be terminated by a tab
	parts := strings.SplitN(trimmedLine, "\t", 2)
	filename = parts[0]
	if len(parts) == 2 {
		// Timestamp is optional, but this header has it.
		ts, err := time.Parse(diffTimeParseLayout, parts[1])
		if err != nil {
			return "", nil, err
		}
		timestamp = &ts
	}

	return filename, timestamp, err
}

// OverflowError is returned when we have overflowed into the start
// of the next file while reading extended headers.
type OverflowError string

func (e OverflowError) Error() string {
	return fmt.Sprintf("overflowed into next file: %s", string(e))
}

// ReadExtendedHeaders reads the extended header lines, if any, from a
// unified diff file (e.g., git's "diff --git a/foo.go b/foo.go", "new
// mode <mode>", "rename from <path>", etc.).
func (r *FileDiffReader) ReadExtendedHeaders() ([]string, error) {
	var xheaders []string
	firstLine := true
	for {
		var line []byte
		if r.fileHeaderLine == nil {
			var err error
			line, err = readLine(r.reader)
			if err == io.EOF {
				return xheaders, &ParseError{r.line, r.offset, ErrExtendedHeadersEOF}
			} else if err != nil {
				return xheaders, err
			}
		} else {
			line = r.fileHeaderLine
			r.fileHeaderLine = nil
		}

		if bytes.HasPrefix(line, []byte("diff --git ")) {
			if firstLine {
				firstLine = false
			} else {
				return xheaders, OverflowError(line)
			}
		}
		if bytes.HasPrefix(line, []byte("--- ")) {
			// We've reached the file header.
			r.fileHeaderLine = line // pass to readOneFileHeader (see fileHeaderLine field doc)
			return xheaders, nil
		}

		// Reached message that file is added/deleted
		if isOnlyInMessage, _, _ := parseOnlyInMessage(line); isOnlyInMessage {
			r.fileHeaderLine = line // pass to readOneFileHeader (see fileHeaderLine field doc)
			return xheaders, nil
		}

		r.line++
		r.offset += int64(len(line))
		xheaders = append(xheaders, string(line))
	}
}

// handleEmpty detects when FileDiff was an empty diff and will not have any hunks
// that follow. It updates fd fields from the parsed extended headers.
func handleEmpty(fd *FileDiff) (wasEmpty bool) {
	var err error
	lineCount := len(fd.Extended)
	if lineCount > 0 && !strings.HasPrefix(fd.Extended[0], "diff --git ") {
		return false
	}
	switch {
	case (lineCount == 3 || lineCount == 4 && strings.HasPrefix(fd.Extended[3], "Binary files ") || lineCount > 4 && strings.HasPrefix(fd.Extended[3], "GIT binary patch")) &&
		strings.HasPrefix(fd.Extended[1], "new file mode "):

		names := strings.SplitN(fd.Extended[0][len("diff --git "):], " ", 2)
		fd.OrigName = "/dev/null"
		fd.NewName, err = strconv.Unquote(names[1])
		if err != nil {
			fd.NewName = names[1]
		}
		return true
	case (lineCount == 3 || lineCount == 4 && strings.HasPrefix(fd.Extended[3], "Binary files ") || lineCount > 4 && strings.HasPrefix(fd.Extended[3], "GIT binary patch")) &&
		strings.HasPrefix(fd.Extended[1], "deleted file mode "):

		names := strings.SplitN(fd.Extended[0][len("diff --git "):], " ", 2)
		fd.OrigName, err = strconv.Unquote(names[0])
		if err != nil {
			fd.OrigName = names[0]
		}
		fd.NewName = "/dev/null"
		return true
	case lineCount == 4 && strings.HasPrefix(fd.Extended[2], "rename from ") && strings.HasPrefix(fd.Extended[3], "rename to "):
		names := strings.SplitN(fd.Extended[0][len("diff --git "):], " ", 2)
		fd.OrigName, err = strconv.Unquote(names[0])
		if err != nil {
			fd.OrigName = names[0]
		}
		fd.NewName, err = strconv.Unquote(names[1])
		if err != nil {
			fd.NewName = names[1]
		}
		return true
	case lineCount == 6 && strings.HasPrefix(fd.Extended[5], "Binary files ") && strings.HasPrefix(fd.Extended[2], "rename from ") && strings.HasPrefix(fd.Extended[3], "rename to "):
		names := strings.SplitN(fd.Extended[0][len("diff --git "):], " ", 2)
		fd.OrigName = names[0]
		fd.NewName = names[1]
		return true
	case lineCount == 3 && strings.HasPrefix(fd.Extended[2], "Binary files ") || lineCount > 3 && strings.HasPrefix(fd.Extended[2], "GIT binary patch"):
		names := strings.SplitN(fd.Extended[0][len("diff --git "):], " ", 2)
		fd.OrigName, err = strconv.Unquote(names[0])
		if err != nil {
			fd.OrigName = names[0]
		}
		fd.NewName, err = strconv.Unquote(names[1])
		if err != nil {
			fd.NewName = names[1]
		}
		return true
	default:
		return false
	}
}

var (
	// ErrNoFileHeader is when a file unified diff has no file header
	// (i.e., the lines that begin with "---" and "+++").
	ErrNoFileHeader = errors.New("expected file header, got EOF")

	// ErrBadFileHeader is when a file unified diff has a malformed
	// file header (i.e., the lines that begin with "---" and "+++").
	ErrBadFileHeader = errors.New("bad file header")

	// ErrExtendedHeadersEOF is when an EOF was encountered while reading extended file headers, which means that there were no ---/+++ headers encountered before hunks (if any) began.
	ErrExtendedHeadersEOF = errors.New("expected file header while reading extended headers, got EOF")

	// ErrBadOnlyInMessage is when a file have a malformed `only in` message
	// Should be in format `Only in {source}: {filename}`
	ErrBadOnlyInMessage = errors.New("bad 'only in' message")
)

// ParseHunks parses hunks from a unified diff. The diff must consist
// only of hunks and not include a file header; if it has a file
// header, use ParseFileDiff.
func ParseHunks(diff []byte) ([]*Hunk, error) {
	r := NewHunksReader(bytes.NewReader(diff))
	hunks, err := r.ReadAllHunks()
	if err != nil {
		return nil, err
	}
	return hunks, nil
}

// NewHunksReader returns a new HunksReader that reads unified diff hunks
// from r.
func NewHunksReader(r io.Reader) *HunksReader {
	return &HunksReader{reader: bufio.NewReader(r)}
}

// A HunksReader reads hunks from a unified diff.
type HunksReader struct {
	line   int
	offset int64
	hunk   *Hunk
	reader *bufio.Reader

	nextHunkHeaderLine []byte
}

// ReadHunk reads one hunk from r. If there are no more hunks, it
// returns error io.EOF.
func (r *HunksReader) ReadHunk() (*Hunk, error) {
	r.hunk = nil
	lastLineFromOrig := true
	var line []byte
	var err error
	for {
		if r.nextHunkHeaderLine != nil {
			// Use stored hunk header line that was scanned in at the
			// completion of the previous hunk's ReadHunk.
			line = r.nextHunkHeaderLine
			r.nextHunkHeaderLine = nil
		} else {
			line, err = readLine(r.reader)
			if err != nil {
				if err == io.EOF && r.hunk != nil {
					return r.hunk, nil
				}
				return nil, err
			}
		}

		// Record position.
		r.line++
		r.offset += int64(len(line))

		if r.hunk == nil {
			// Check for presence of hunk header.
			if !bytes.HasPrefix(line, hunkPrefix) {
				return nil, &ParseError{r.line, r.offset, ErrNoHunkHeader}
			}

			// Parse hunk header.
			r.hunk = &Hunk{}
			items := []interface{}{
				&r.hunk.OrigStartLine, &r.hunk.OrigLines,
				&r.hunk.NewStartLine, &r.hunk.NewLines,
			}
			header, section, err := normalizeHeader(string(line))
			if err != nil {
				return nil, &ParseError{r.line, r.offset, err}
			}
			n, err := fmt.Sscanf(header, hunkHeader, items...)
			if err != nil {
				return nil, err
			}
			if n < len(items) {
				return nil, &ParseError{r.line, r.offset, &ErrBadHunkHeader{header: string(line)}}
			}

			r.hunk.Section = section
		} else {
			// Read hunk body line.

			// If the line starts with `---` and the next one with `+++` we're
			// looking at a non-extended file header and need to abort.
			if bytes.HasPrefix(line, []byte("---")) {
				ok, err := peekPrefix(r.reader, "+++")
				if err != nil {
					return r.hunk, err
				}
				if ok {
					return r.hunk, &ParseError{r.line, r.offset, &ErrBadHunkLine{Line: line}}
				}
			}

			// If the line starts with the hunk prefix, this hunk is complete.
			if bytes.HasPrefix(line, hunkPrefix) {
				// But we've already read in the next hunk's
				// header, so we need to be sure that the next call to
				// ReadHunk starts with that header.
				r.nextHunkHeaderLine = line

				// Rewind position.
				r.line--
				r.offset -= int64(len(line))

				return r.hunk, nil
			}

			if len(line) >= 1 && !linePrefix(line[0]) {
				// Bad hunk header line. If we're reading a multi-file
				// diff, this may be the end of the current
				// file. Return a "rich" error that lets our caller
				// handle that case.
				return r.hunk, &ParseError{r.line, r.offset, &ErrBadHunkLine{Line: line}}
			}
			if bytes.Equal(line, []byte(noNewlineMessage)) {
				if lastLineFromOrig {
					// Retain the newline in the body (otherwise the
					// diff line would be like "-a+b", where "+b" is
					// the the next line of the new file, which is not
					// validly formatted) but record that the orig had
					// no newline.
					r.hunk.OrigNoNewlineAt = int32(len(r.hunk.Body))
				} else {
					// Remove previous line's newline.
					if len(r.hunk.Body) != 0 {
						r.hunk.Body = r.hunk.Body[:len(r.hunk.Body)-1]
					}
				}
				continue
			}

			if len(line) > 0 {
				lastLineFromOrig = line[0] == '-'
			}

			r.hunk.Body = append(r.hunk.Body, line...)
			r.hunk.Body = append(r.hunk.Body, '\n')
		}
	}
}

const noNewlineMessage = `\ No newline at end of file`

// linePrefixes is the set of all characters a valid line in a diff
// hunk can start with. '\' can appear in diffs when no newline is
// present at the end of a file.
// See: 'http://www.gnu.org/software/diffutils/manual/diffutils.html#Incomplete-Lines'
var linePrefixes = []byte{' ', '-', '+', '\\'}

// linePrefix returns true if 'c' is in 'linePrefixes'.
func linePrefix(c byte) bool {
	for _, p := range linePrefixes {
		if p == c {
			return true
		}
	}
	return false
}

// peekPrefix peeks into the given reader to check whether the next
// bytes match the given prefix.
func peekPrefix(reader *bufio.Reader, prefix string) (bool, error) {
	next, err := reader.Peek(len(prefix))
	if err != nil {
		if err == io.EOF {
			return false, nil
		}
		return false, err
	}
	return bytes.HasPrefix(next, []byte(prefix)), nil
}

// normalizeHeader takes a header of the form:
// "@@ -linestart[,chunksize] +linestart[,chunksize] @@ section"
// and returns two strings, with the first in the form:
// "@@ -linestart,chunksize +linestart,chunksize @@".
// where linestart and chunksize are both integers. The second is the
// optional section header. chunksize may be omitted from the header
// if its value is 1. normalizeHeader returns an error if the header
// is not in the correct format.
func normalizeHeader(header string) (string, string, error) {
	// Split the header into five parts: the first '@@', the two
	// ranges, the last '@@', and the optional section.
	pieces := strings.SplitN(header, " ", 5)
	if len(pieces) < 4 {
		return "", "", &ErrBadHunkHeader{header: header}
	}

	if pieces[0] != "@@" {
		return "", "", &ErrBadHunkHeader{header: header}
	}
	for i := 1; i < 3; i++ {
		if !strings.ContainsRune(pieces[i], ',') {
			pieces[i] = pieces[i] + ",1"
		}
	}
	if pieces[3] != "@@" {
		return "", "", &ErrBadHunkHeader{header: header}
	}

	var section string
	if len(pieces) == 5 {
		section = pieces[4]
	}
	return strings.Join(pieces, " "), strings.TrimSpace(section), nil
}

// ReadAllHunks reads all remaining hunks from r. A successful call
// returns err == nil, not err == EOF. Because ReadAllHunks is defined
// to read until EOF, it does not treat end of file as an error to be
// reported.
func (r *HunksReader) ReadAllHunks() ([]*Hunk, error) {
	var hunks []*Hunk
	linesRead := int32(0)
	for {
		hunk, err := r.ReadHunk()
		if err == io.EOF {
			return hunks, nil
		}
		if hunk != nil {
			linesRead++ // account for the hunk header line
			hunk.StartPosition = linesRead
			hunks = append(hunks, hunk)
			linesRead += int32(bytes.Count(hunk.Body, []byte{'\n'}))
		}
		if err != nil {
			return hunks, err
		}
	}
}

// parseOnlyInMessage checks if line is a "Only in {source}: {filename}" and returns source and filename
func parseOnlyInMessage(line []byte) (bool, []byte, []byte) {
	if !bytes.HasPrefix(line, onlyInMessagePrefix) {
		return false, nil, nil
	}
	line = line[len(onlyInMessagePrefix):]
	idx := bytes.Index(line, []byte(": "))
	if idx < 0 {
		return false, nil, nil
	}
	return true, line[:idx], line[idx+2:]
}

// A ParseError is a description of a unified diff syntax error.
type ParseError struct {
	Line   int   // Line where the error occurred
	Offset int64 // Offset where the error occurred
	Err    error // The actual error
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("line %d, char %d: %s", e.Line, e.Offset, e.Err)
}

// ErrNoHunkHeader indicates that a unified diff hunk header was
// expected but not found during parsing.
var ErrNoHunkHeader = errors.New("no hunk header")

// ErrBadHunkHeader indicates that a malformed unified diff hunk
// header was encountered during parsing.
type ErrBadHunkHeader struct {
	header string
}

func (e *ErrBadHunkHeader) Error() string {
	if e.header == "" {
		return "bad hunk header"
	}
	return "bad hunk header: " + e.header
}

// ErrBadHunkLine is when a line not beginning with ' ', '-', '+', or
// '\' is encountered while reading a hunk. In the context of reading
// a single hunk or file, it is an unexpected error. In a multi-file
// diff, however, it indicates that the current file's diff is
// complete (and remaining diff data will describe another file
// unified diff).
type ErrBadHunkLine struct {
	Line []byte
}

func (e *ErrBadHunkLine) Error() string {
	m := "bad hunk line (does not start with ' ', '-', '+', or '\\')"
	if len(e.Line) == 0 {
		return m
	}
	return m + ": " + string(e.Line)
}
