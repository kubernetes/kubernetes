package opts

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"
)

func tmpFileWithContent(content string, t *testing.T) string {
	tmpFile, err := ioutil.TempFile("", "envfile-test")
	if err != nil {
		t.Fatal(err)
	}
	defer tmpFile.Close()

	tmpFile.WriteString(content)
	return tmpFile.Name()
}

// Test ParseEnvFile for a file with a few well formatted lines
func TestParseEnvFileGoodFile(t *testing.T) {
	content := `foo=bar
    baz=quux
# comment

_foobar=foobaz
with.dots=working
and_underscore=working too
`
	// Adding a newline + a line with pure whitespace.
	// This is being done like this instead of the block above
	// because it's common for editors to trim trailing whitespace
	// from lines, which becomes annoying since that's the
	// exact thing we need to test.
	content += "\n    \t  "
	tmpFile := tmpFileWithContent(content, t)
	defer os.Remove(tmpFile)

	lines, err := ParseEnvFile(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	expectedLines := []string{
		"foo=bar",
		"baz=quux",
		"_foobar=foobaz",
		"with.dots=working",
		"and_underscore=working too",
	}

	if !reflect.DeepEqual(lines, expectedLines) {
		t.Fatal("lines not equal to expected_lines")
	}
}

// Test ParseEnvFile for an empty file
func TestParseEnvFileEmptyFile(t *testing.T) {
	tmpFile := tmpFileWithContent("", t)
	defer os.Remove(tmpFile)

	lines, err := ParseEnvFile(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	if len(lines) != 0 {
		t.Fatal("lines not empty; expected empty")
	}
}

// Test ParseEnvFile for a non existent file
func TestParseEnvFileNonExistentFile(t *testing.T) {
	_, err := ParseEnvFile("foo_bar_baz")
	if err == nil {
		t.Fatal("ParseEnvFile succeeded; expected failure")
	}
	if _, ok := err.(*os.PathError); !ok {
		t.Fatalf("Expected a PathError, got [%v]", err)
	}
}

// Test ParseEnvFile for a badly formatted file
func TestParseEnvFileBadlyFormattedFile(t *testing.T) {
	content := `foo=bar
    f   =quux
`

	tmpFile := tmpFileWithContent(content, t)
	defer os.Remove(tmpFile)

	_, err := ParseEnvFile(tmpFile)
	if err == nil {
		t.Fatalf("Expected a ErrBadEnvVariable, got nothing")
	}
	if _, ok := err.(ErrBadEnvVariable); !ok {
		t.Fatalf("Expected a ErrBadEnvVariable, got [%v]", err)
	}
	expectedMessage := "poorly formatted environment: variable 'f   ' has white spaces"
	if err.Error() != expectedMessage {
		t.Fatalf("Expected [%v], got [%v]", expectedMessage, err.Error())
	}
}

// Test ParseEnvFile for a file with a line exceeding bufio.MaxScanTokenSize
func TestParseEnvFileLineTooLongFile(t *testing.T) {
	content := strings.Repeat("a", bufio.MaxScanTokenSize+42)
	content = fmt.Sprint("foo=", content)

	tmpFile := tmpFileWithContent(content, t)
	defer os.Remove(tmpFile)

	_, err := ParseEnvFile(tmpFile)
	if err == nil {
		t.Fatal("ParseEnvFile succeeded; expected failure")
	}
}

// ParseEnvFile with a random file, pass through
func TestParseEnvFileRandomFile(t *testing.T) {
	content := `first line
another invalid line`
	tmpFile := tmpFileWithContent(content, t)
	defer os.Remove(tmpFile)

	_, err := ParseEnvFile(tmpFile)

	if err == nil {
		t.Fatalf("Expected a ErrBadEnvVariable, got nothing")
	}
	if _, ok := err.(ErrBadEnvVariable); !ok {
		t.Fatalf("Expected a ErrBadEnvvariable, got [%v]", err)
	}
	expectedMessage := "poorly formatted environment: variable 'first line' has white spaces"
	if err.Error() != expectedMessage {
		t.Fatalf("Expected [%v], got [%v]", expectedMessage, err.Error())
	}
}
