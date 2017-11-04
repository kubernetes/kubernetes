package gopass

import (
	"errors"
	"fmt"
	"io"
	"os"
)

type FdReader interface {
	io.Reader
	Fd() uintptr
}

var defaultGetCh = func(r io.Reader) (byte, error) {
	buf := make([]byte, 1)
	if n, err := r.Read(buf); n == 0 || err != nil {
		if err != nil {
			return 0, err
		}
		return 0, io.EOF
	}
	return buf[0], nil
}

var (
	maxLength            = 512
	ErrInterrupted       = errors.New("interrupted")
	ErrMaxLengthExceeded = fmt.Errorf("maximum byte limit (%v) exceeded", maxLength)

	// Provide variable so that tests can provide a mock implementation.
	getch = defaultGetCh
)

// getPasswd returns the input read from terminal.
// If prompt is not empty, it will be output as a prompt to the user
// If masked is true, typing will be matched by asterisks on the screen.
// Otherwise, typing will echo nothing.
func getPasswd(prompt string, masked bool, r FdReader, w io.Writer) ([]byte, error) {
	var err error
	var pass, bs, mask []byte
	if masked {
		bs = []byte("\b \b")
		mask = []byte("*")
	}

	if isTerminal(r.Fd()) {
		if oldState, err := makeRaw(r.Fd()); err != nil {
			return pass, err
		} else {
			defer func() {
				restore(r.Fd(), oldState)
				fmt.Fprintln(w)
			}()
		}
	}

	if prompt != "" {
		fmt.Fprint(w, prompt)
	}

	// Track total bytes read, not just bytes in the password.  This ensures any
	// errors that might flood the console with nil or -1 bytes infinitely are
	// capped.
	var counter int
	for counter = 0; counter <= maxLength; counter++ {
		if v, e := getch(r); e != nil {
			err = e
			break
		} else if v == 127 || v == 8 {
			if l := len(pass); l > 0 {
				pass = pass[:l-1]
				fmt.Fprint(w, string(bs))
			}
		} else if v == 13 || v == 10 {
			break
		} else if v == 3 {
			err = ErrInterrupted
			break
		} else if v != 0 {
			pass = append(pass, v)
			fmt.Fprint(w, string(mask))
		}
	}

	if counter > maxLength {
		err = ErrMaxLengthExceeded
	}

	return pass, err
}

// GetPasswd returns the password read from the terminal without echoing input.
// The returned byte array does not include end-of-line characters.
func GetPasswd() ([]byte, error) {
	return getPasswd("", false, os.Stdin, os.Stdout)
}

// GetPasswdMasked returns the password read from the terminal, echoing asterisks.
// The returned byte array does not include end-of-line characters.
func GetPasswdMasked() ([]byte, error) {
	return getPasswd("", true, os.Stdin, os.Stdout)
}

// GetPasswdPrompt prompts the user and returns the password read from the terminal.
// If mask is true, then asterisks are echoed.
// The returned byte array does not include end-of-line characters.
func GetPasswdPrompt(prompt string, mask bool, r FdReader, w io.Writer) ([]byte, error) {
	return getPasswd(prompt, mask, r, w)
}
