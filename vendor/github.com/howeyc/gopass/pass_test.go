package gopass

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"testing"
	"time"
)

// TestGetPasswd tests the password creation and output based on a byte buffer
// as input to mock the underlying getch() methods.
func TestGetPasswd(t *testing.T) {
	type testData struct {
		input []byte

		// Due to how backspaces are written, it is easier to manually write
		// each expected output for the masked cases.
		masked   string
		password string
		byesLeft int
		reason   string
	}

	ds := []testData{
		testData{[]byte("abc\n"), "***", "abc", 0, "Password parsing should stop at \\n"},
		testData{[]byte("abc\r"), "***", "abc", 0, "Password parsing should stop at \\r"},
		testData{[]byte("a\nbc\n"), "*", "a", 3, "Password parsing should stop at \\n"},
		testData{[]byte("*!]|\n"), "****", "*!]|", 0, "Special characters shouldn't affect the password."},

		testData{[]byte("abc\r\n"), "***", "abc", 1,
			"Password parsing should stop at \\r; Windows LINE_MODE should be unset so \\r is not converted to \\r\\n."},

		testData{[]byte{'a', 'b', 'c', 8, '\n'}, "***\b \b", "ab", 0, "Backspace byte should remove the last read byte."},
		testData{[]byte{'a', 'b', 127, 'c', '\n'}, "**\b \b*", "ac", 0, "Delete byte should remove the last read byte."},
		testData{[]byte{'a', 'b', 127, 'c', 8, 127, '\n'}, "**\b \b*\b \b\b \b", "", 0, "Successive deletes continue to delete."},
		testData{[]byte{8, 8, 8, '\n'}, "", "", 0, "Deletes before characters are noops."},
		testData{[]byte{8, 8, 8, 'a', 'b', 'c', '\n'}, "***", "abc", 0, "Deletes before characters are noops."},

		testData{[]byte{'a', 'b', 0, 'c', '\n'}, "***", "abc", 0,
			"Nil byte should be ignored due; may get unintended nil bytes from syscalls on Windows."},
	}

	// Redirecting output for tests as they print to os.Stdout but we want to
	// capture and test the output.
	for _, masked := range []bool{true, false} {
		for _, d := range ds {
			pipeBytesToStdin(d.input)

			r, w, err := os.Pipe()
			if err != nil {
				t.Fatal(err.Error())
			}

			result, err := getPasswd("", masked, os.Stdin, w)
			if err != nil {
				t.Errorf("Error getting password: %s", err.Error())
			}
			leftOnBuffer := flushStdin()

			// Test output (masked and unmasked).  Delete/backspace actually
			// deletes, overwrites and deletes again.  As a result, we need to
			// remove those from the pipe afterwards to mimic the console's
			// interpretation of those bytes.
			w.Close()
			output, err := ioutil.ReadAll(r)
			if err != nil {
				t.Fatal(err.Error())
			}
			var expectedOutput []byte
			if masked {
				expectedOutput = []byte(d.masked)
			} else {
				expectedOutput = []byte("")
			}
			if bytes.Compare(expectedOutput, output) != 0 {
				t.Errorf("Expected output to equal %v (%q) but got %v (%q) instead when masked=%v. %s", expectedOutput, string(expectedOutput), output, string(output), masked, d.reason)
			}

			if string(result) != d.password {
				t.Errorf("Expected %q but got %q instead when masked=%v. %s", d.password, result, masked, d.reason)
			}

			if leftOnBuffer != d.byesLeft {
				t.Errorf("Expected %v bytes left on buffer but instead got %v when masked=%v. %s", d.byesLeft, leftOnBuffer, masked, d.reason)
			}
		}
	}
}

// TestPipe ensures we get our expected pipe behavior.
func TestPipe(t *testing.T) {
	type testData struct {
		input    string
		password string
		expError error
	}
	ds := []testData{
		testData{"abc", "abc", io.EOF},
		testData{"abc\n", "abc", nil},
		testData{"abc\r", "abc", nil},
		testData{"abc\r\n", "abc", nil},
	}

	for _, d := range ds {
		_, err := pipeToStdin(d.input)
		if err != nil {
			t.Log("Error writing input to stdin:", err)
			t.FailNow()
		}
		pass, err := GetPasswd()
		if string(pass) != d.password {
			t.Errorf("Expected %q but got %q instead.", d.password, string(pass))
		}
		if err != d.expError {
			t.Errorf("Expected %v but got %q instead.", d.expError, err)
		}
	}
}

// flushStdin reads from stdin for .5 seconds to ensure no bytes are left on
// the buffer.  Returns the number of bytes read.
func flushStdin() int {
	ch := make(chan byte)
	go func(ch chan byte) {
		reader := bufio.NewReader(os.Stdin)
		for {
			b, err := reader.ReadByte()
			if err != nil { // Maybe log non io.EOF errors, if you want
				close(ch)
				return
			}
			ch <- b
		}
		close(ch)
	}(ch)

	numBytes := 0
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return numBytes
			}
			numBytes++
		case <-time.After(500 * time.Millisecond):
			return numBytes
		}
	}
	return numBytes
}

// pipeToStdin pipes the given string onto os.Stdin by replacing it with an
// os.Pipe.  The write end of the pipe is closed so that EOF is read after the
// final byte.
func pipeToStdin(s string) (int, error) {
	pipeReader, pipeWriter, err := os.Pipe()
	if err != nil {
		fmt.Println("Error getting os pipes:", err)
		os.Exit(1)
	}
	os.Stdin = pipeReader
	w, err := pipeWriter.WriteString(s)
	pipeWriter.Close()
	return w, err
}

func pipeBytesToStdin(b []byte) (int, error) {
	return pipeToStdin(string(b))
}

// TestGetPasswd_Err tests errors are properly handled from getch()
func TestGetPasswd_Err(t *testing.T) {
	var inBuffer *bytes.Buffer
	getch = func(io.Reader) (byte, error) {
		b, err := inBuffer.ReadByte()
		if err != nil {
			return 13, err
		}
		if b == 'z' {
			return 'z', fmt.Errorf("Forced error; byte returned should not be considered accurate.")
		}
		return b, nil
	}
	defer func() { getch = defaultGetCh }()

	for input, expectedPassword := range map[string]string{"abc": "abc", "abzc": "ab"} {
		inBuffer = bytes.NewBufferString(input)
		p, err := GetPasswdMasked()
		if string(p) != expectedPassword {
			t.Errorf("Expected %q but got %q instead.", expectedPassword, p)
		}
		if err == nil {
			t.Errorf("Expected error to be returned.")
		}
	}
}

func TestMaxPasswordLength(t *testing.T) {
	type testData struct {
		input       []byte
		expectedErr error

		// Helper field to output in case of failure; rather than hundreds of
		// bytes.
		inputDesc string
	}

	ds := []testData{
		testData{append(bytes.Repeat([]byte{'a'}, maxLength), '\n'), nil, fmt.Sprintf("%v 'a' bytes followed by a newline", maxLength)},
		testData{append(bytes.Repeat([]byte{'a'}, maxLength+1), '\n'), ErrMaxLengthExceeded, fmt.Sprintf("%v 'a' bytes followed by a newline", maxLength+1)},
		testData{append(bytes.Repeat([]byte{0x00}, maxLength+1), '\n'), ErrMaxLengthExceeded, fmt.Sprintf("%v 0x00 bytes followed by a newline", maxLength+1)},
	}

	for _, d := range ds {
		pipeBytesToStdin(d.input)
		_, err := GetPasswd()
		if err != d.expectedErr {
			t.Errorf("Expected error to be %v; isntead got %v from %v", d.expectedErr, err, d.inputDesc)
		}
	}
}
