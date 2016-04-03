package winconsole

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
)

const (
	WRITE_OPERATION   = iota
	COMMAND_OPERATION = iota
)

var languages = []string{
	"Български",
	"Català",
	"Čeština",
	"Ελληνικά",
	"Español",
	"Esperanto",
	"Euskara",
	"Français",
	"Galego",
	"한국어",
	"ქართული",
	"Latviešu",
	"Lietuvių",
	"Magyar",
	"Nederlands",
	"日本語",
	"Norsk bokmål",
	"Norsk nynorsk",
	"Polski",
	"Português",
	"Română",
	"Русский",
	"Slovenčina",
	"Slovenščina",
	"Српски",
	"српскохрватски",
	"Suomi",
	"Svenska",
	"ไทย",
	"Tiếng Việt",
	"Türkçe",
	"Українська",
	"中文",
}

// Mock terminal handler object
type mockTerminal struct {
	OutputCommandSequence []terminalOperation
}

// Used for recording the callback data
type terminalOperation struct {
	Operation int
	Data      []byte
	Str       string
}

func (mt *mockTerminal) record(operation int, data []byte) {
	op := terminalOperation{
		Operation: operation,
		Data:      make([]byte, len(data)),
	}
	copy(op.Data, data)
	op.Str = string(op.Data)
	mt.OutputCommandSequence = append(mt.OutputCommandSequence, op)
}

func (mt *mockTerminal) HandleOutputCommand(fd uintptr, command []byte) (n int, err error) {
	mt.record(COMMAND_OPERATION, command)
	return len(command), nil
}

func (mt *mockTerminal) HandleInputSequence(fd uintptr, command []byte) (n int, err error) {
	return 0, nil
}

func (mt *mockTerminal) WriteChars(fd uintptr, w io.Writer, p []byte) (n int, err error) {
	mt.record(WRITE_OPERATION, p)
	return len(p), nil
}

func (mt *mockTerminal) ReadChars(fd uintptr, w io.Reader, p []byte) (n int, err error) {
	return len(p), nil
}

func assertTrue(t *testing.T, cond bool, format string, args ...interface{}) {
	if !cond {
		t.Errorf(format, args...)
	}
}

// reflect.DeepEqual does not provide detailed information as to what excatly failed.
func assertBytesEqual(t *testing.T, expected, actual []byte, format string, args ...interface{}) {
	match := true
	mismatchIndex := 0
	if len(expected) == len(actual) {
		for i := 0; i < len(expected); i++ {
			if expected[i] != actual[i] {
				match = false
				mismatchIndex = i
				break
			}
		}
	} else {
		match = false
		t.Errorf("Lengths don't match Expected=%d Actual=%d", len(expected), len(actual))
	}
	if !match {
		t.Errorf("Mismatch at index %d ", mismatchIndex)
		t.Errorf("\tActual String   = %s", string(actual))
		t.Errorf("\tExpected String = %s", string(expected))
		t.Errorf("\tActual          = %v", actual)
		t.Errorf("\tExpected        = %v", expected)
		t.Errorf(format, args)
	}
}

// Just to make sure :)
func TestAssertEqualBytes(t *testing.T) {
	data := []byte{9, 9, 1, 1, 1, 9, 9}
	assertBytesEqual(t, data, data, "Self")
	assertBytesEqual(t, data[1:4], data[1:4], "Self")
	assertBytesEqual(t, []byte{1, 1}, []byte{1, 1}, "Simple match")
	assertBytesEqual(t, []byte{1, 2, 3}, []byte{1, 2, 3}, "content mismatch")
	assertBytesEqual(t, []byte{1, 1, 1}, data[2:5], "slice match")
}

/*
func TestAssertEqualBytesNegative(t *testing.T) {
	AssertBytesEqual(t, []byte{1, 1}, []byte{1}, "Length mismatch")
	AssertBytesEqual(t, []byte{1, 1}, []byte{1}, "Length mismatch")
	AssertBytesEqual(t, []byte{1, 2, 3}, []byte{1, 1, 1}, "content mismatch")
}*/

// Checks that the calls received
func assertHandlerOutput(t *testing.T, mock *mockTerminal, plainText string, commands ...string) {
	text := make([]byte, 0, 3*len(plainText))
	cmdIndex := 0
	for opIndex := 0; opIndex < len(mock.OutputCommandSequence); opIndex++ {
		op := mock.OutputCommandSequence[opIndex]
		if op.Operation == WRITE_OPERATION {
			t.Logf("\nThe data is[%d] == %s", opIndex, string(op.Data))
			text = append(text[:], op.Data...)
		} else {
			assertTrue(t, mock.OutputCommandSequence[opIndex].Operation == COMMAND_OPERATION, "Operation should be command : %s", fmt.Sprintf("%+v", mock))
			assertBytesEqual(t, StringToBytes(commands[cmdIndex]), mock.OutputCommandSequence[opIndex].Data, "Command data should match")
			cmdIndex++
		}
	}
	assertBytesEqual(t, StringToBytes(plainText), text, "Command data should match %#v", mock)
}

func StringToBytes(str string) []byte {
	bytes := make([]byte, len(str))
	copy(bytes[:], str)
	return bytes
}

func TestParseAnsiCommand(t *testing.T) {
	// Note: if the parameter does not exist then the empty value is returned

	c := parseAnsiCommand(StringToBytes("\x1Bm"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "" == c.getParam(0), "should return empty string")
	assertTrue(t, "" == c.getParam(1), "should return empty string")

	// Escape sequence - ESC[
	c = parseAnsiCommand(StringToBytes("\x1B[m"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "" == c.getParam(0), "should return empty string")
	assertTrue(t, "" == c.getParam(1), "should return empty string")

	// Escape sequence With empty parameters- ESC[
	c = parseAnsiCommand(StringToBytes("\x1B[;m"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "" == c.getParam(0), "should return empty string")
	assertTrue(t, "" == c.getParam(1), "should return empty string")
	assertTrue(t, "" == c.getParam(2), "should return empty string")

	// Escape sequence With empty muliple parameters- ESC[
	c = parseAnsiCommand(StringToBytes("\x1B[;;m"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "" == c.getParam(0), "")
	assertTrue(t, "" == c.getParam(1), "")
	assertTrue(t, "" == c.getParam(2), "")

	// Escape sequence With muliple parameters- ESC[
	c = parseAnsiCommand(StringToBytes("\x1B[1;2;3m"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "1" == c.getParam(0), "")
	assertTrue(t, "2" == c.getParam(1), "")
	assertTrue(t, "3" == c.getParam(2), "")

	// Escape sequence With muliple parameters- some missing
	c = parseAnsiCommand(StringToBytes("\x1B[1;;3;;;6m"))
	assertTrue(t, c.Command == "m", "Command should be m")
	assertTrue(t, "1" == c.getParam(0), "")
	assertTrue(t, "" == c.getParam(1), "")
	assertTrue(t, "3" == c.getParam(2), "")
	assertTrue(t, "" == c.getParam(3), "")
	assertTrue(t, "" == c.getParam(4), "")
	assertTrue(t, "6" == c.getParam(5), "")
}

func newBufferedMockTerm() (stdOut io.Writer, stdErr io.Writer, stdIn io.ReadCloser, mock *mockTerminal) {
	var input bytes.Buffer
	var output bytes.Buffer
	var err bytes.Buffer

	mock = &mockTerminal{
		OutputCommandSequence: make([]terminalOperation, 0, 256),
	}

	stdOut = &terminalWriter{
		wrappedWriter: &output,
		emulator:      mock,
		command:       make([]byte, 0, 256),
	}
	stdErr = &terminalWriter{
		wrappedWriter: &err,
		emulator:      mock,
		command:       make([]byte, 0, 256),
	}
	stdIn = &terminalReader{
		wrappedReader: ioutil.NopCloser(&input),
		emulator:      mock,
		command:       make([]byte, 0, 256),
	}

	return
}

func TestOutputSimple(t *testing.T) {
	stdOut, _, _, mock := newBufferedMockTerm()

	stdOut.Write(StringToBytes("Hello world"))
	stdOut.Write(StringToBytes("\x1BmHello again"))

	assertTrue(t, mock.OutputCommandSequence[0].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello world"), mock.OutputCommandSequence[0].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[1].Operation == COMMAND_OPERATION, "Operation should be command : %+v", mock)
	assertBytesEqual(t, StringToBytes("\x1Bm"), mock.OutputCommandSequence[1].Data, "Command data should match")

	assertTrue(t, mock.OutputCommandSequence[2].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello again"), mock.OutputCommandSequence[2].Data, "Write data should match")
}

func TestOutputSplitCommand(t *testing.T) {
	stdOut, _, _, mock := newBufferedMockTerm()

	stdOut.Write(StringToBytes("Hello world\x1B[1;2;3"))
	stdOut.Write(StringToBytes("mHello again"))

	assertTrue(t, mock.OutputCommandSequence[0].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello world"), mock.OutputCommandSequence[0].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[1].Operation == COMMAND_OPERATION, "Operation should be command : %+v", mock)
	assertBytesEqual(t, StringToBytes("\x1B[1;2;3m"), mock.OutputCommandSequence[1].Data, "Command data should match")

	assertTrue(t, mock.OutputCommandSequence[2].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello again"), mock.OutputCommandSequence[2].Data, "Write data should match")
}

func TestOutputMultipleCommands(t *testing.T) {
	stdOut, _, _, mock := newBufferedMockTerm()

	stdOut.Write(StringToBytes("Hello world"))
	stdOut.Write(StringToBytes("\x1B[1;2;3m"))
	stdOut.Write(StringToBytes("\x1B[J"))
	stdOut.Write(StringToBytes("Hello again"))

	assertTrue(t, mock.OutputCommandSequence[0].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello world"), mock.OutputCommandSequence[0].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[1].Operation == COMMAND_OPERATION, "Operation should be command : %+v", mock)
	assertBytesEqual(t, StringToBytes("\x1B[1;2;3m"), mock.OutputCommandSequence[1].Data, "Command data should match")

	assertTrue(t, mock.OutputCommandSequence[2].Operation == COMMAND_OPERATION, "Operation should be command : %+v", mock)
	assertBytesEqual(t, StringToBytes("\x1B[J"), mock.OutputCommandSequence[2].Data, "Command data should match")

	assertTrue(t, mock.OutputCommandSequence[3].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, StringToBytes("Hello again"), mock.OutputCommandSequence[3].Data, "Write data should match")
}

// Splits the given data in two chunks , makes two writes and checks the split data is parsed correctly
// checks output write/command is passed to handler correctly
func helpsTestOutputSplitChunksAtIndex(t *testing.T, i int, data []byte) {
	t.Logf("\ni=%d", i)
	stdOut, _, _, mock := newBufferedMockTerm()

	t.Logf("\nWriting chunk[0] == %s", string(data[:i]))
	t.Logf("\nWriting chunk[1] == %s", string(data[i:]))
	stdOut.Write(data[:i])
	stdOut.Write(data[i:])

	assertTrue(t, mock.OutputCommandSequence[0].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, data[:i], mock.OutputCommandSequence[0].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[1].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, data[i:], mock.OutputCommandSequence[1].Data, "Write data should match")
}

// Splits the given data in three chunks , makes three writes and checks the split data is parsed correctly
// checks output write/command is passed to handler correctly
func helpsTestOutputSplitThreeChunksAtIndex(t *testing.T, data []byte, i int, j int) {
	stdOut, _, _, mock := newBufferedMockTerm()

	t.Logf("\nWriting chunk[0] == %s", string(data[:i]))
	t.Logf("\nWriting chunk[1] == %s", string(data[i:j]))
	t.Logf("\nWriting chunk[2] == %s", string(data[j:]))
	stdOut.Write(data[:i])
	stdOut.Write(data[i:j])
	stdOut.Write(data[j:])

	assertTrue(t, mock.OutputCommandSequence[0].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, data[:i], mock.OutputCommandSequence[0].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[1].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, data[i:j], mock.OutputCommandSequence[1].Data, "Write data should match")

	assertTrue(t, mock.OutputCommandSequence[2].Operation == WRITE_OPERATION, "Operation should be Write : %#v", mock)
	assertBytesEqual(t, data[j:], mock.OutputCommandSequence[2].Data, "Write data should match")
}

// Splits the output into two parts and tests all such possible pairs
func helpsTestOutputSplitChunks(t *testing.T, data []byte) {
	for i := 1; i < len(data)-1; i++ {
		helpsTestOutputSplitChunksAtIndex(t, i, data)
	}
}

// Splits the output in three parts and tests all such possible triples
func helpsTestOutputSplitThreeChunks(t *testing.T, data []byte) {
	for i := 1; i < len(data)-2; i++ {
		for j := i + 1; j < len(data)-1; j++ {
			helpsTestOutputSplitThreeChunksAtIndex(t, data, i, j)
		}
	}
}

func helpsTestOutputSplitCommandsAtIndex(t *testing.T, data []byte, i int, plainText string, commands ...string) {
	t.Logf("\ni=%d", i)
	stdOut, _, _, mock := newBufferedMockTerm()

	stdOut.Write(data[:i])
	stdOut.Write(data[i:])
	assertHandlerOutput(t, mock, plainText, commands...)
}

func helpsTestOutputSplitCommands(t *testing.T, data []byte, plainText string, commands ...string) {
	for i := 1; i < len(data)-1; i++ {
		helpsTestOutputSplitCommandsAtIndex(t, data, i, plainText, commands...)
	}
}

func injectCommandAt(data string, i int, command string) string {
	retValue := make([]byte, len(data)+len(command)+4)
	retValue = append(retValue, data[:i]...)
	retValue = append(retValue, data[i:]...)
	return string(retValue)
}

func TestOutputSplitChunks(t *testing.T) {
	data := StringToBytes("qwertyuiopasdfghjklzxcvbnm")
	helpsTestOutputSplitChunks(t, data)
	helpsTestOutputSplitChunks(t, StringToBytes("BBBBB"))
	helpsTestOutputSplitThreeChunks(t, StringToBytes("ABCDE"))
}

func TestOutputSplitChunksIncludingCommands(t *testing.T) {
	helpsTestOutputSplitCommands(t, StringToBytes("Hello world.\x1B[mHello again."), "Hello world.Hello again.", "\x1B[m")
	helpsTestOutputSplitCommandsAtIndex(t, StringToBytes("Hello world.\x1B[mHello again."), 2, "Hello world.Hello again.", "\x1B[m")
}

func TestSplitChunkUnicode(t *testing.T) {
	for _, l := range languages {
		data := StringToBytes(l)
		helpsTestOutputSplitChunks(t, data)
		helpsTestOutputSplitThreeChunks(t, data)
	}
}
