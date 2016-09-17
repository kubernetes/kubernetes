package cli

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"github.com/bgentry/speakeasy"
	"github.com/mattn/go-isatty"
)

// Ui is an interface for interacting with the terminal, or "interface"
// of a CLI. This abstraction doesn't have to be used, but helps provide
// a simple, layerable way to manage user interactions.
type Ui interface {
	// Ask asks the user for input using the given query. The response is
	// returned as the given string, or an error.
	Ask(string) (string, error)

	// AskSecret asks the user for input using the given query, but does not echo
	// the keystrokes to the terminal.
	AskSecret(string) (string, error)

	// Output is called for normal standard output.
	Output(string)

	// Info is called for information related to the previous output.
	// In general this may be the exact same as Output, but this gives
	// Ui implementors some flexibility with output formats.
	Info(string)

	// Error is used for any error messages that might appear on standard
	// error.
	Error(string)

	// Warn is used for any warning messages that might appear on standard
	// error.
	Warn(string)
}

// BasicUi is an implementation of Ui that just outputs to the given
// writer. This UI is not threadsafe by default, but you can wrap it
// in a ConcurrentUi to make it safe.
type BasicUi struct {
	Reader      io.Reader
	Writer      io.Writer
	ErrorWriter io.Writer
}

func (u *BasicUi) Ask(query string) (string, error) {
	return u.ask(query, false)
}

func (u *BasicUi) AskSecret(query string) (string, error) {
	return u.ask(query, true)
}

func (u *BasicUi) ask(query string, secret bool) (string, error) {
	if _, err := fmt.Fprint(u.Writer, query+" "); err != nil {
		return "", err
	}

	// Register for interrupts so that we can catch it and immediately
	// return...
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	defer signal.Stop(sigCh)

	// Ask for input in a go-routine so that we can ignore it.
	errCh := make(chan error, 1)
	lineCh := make(chan string, 1)
	go func() {
		var line string
		var err error
		if secret && isatty.IsTerminal(os.Stdin.Fd()) {
			line, err = speakeasy.Ask("")
		} else {
			r := bufio.NewReader(u.Reader)
			line, err = r.ReadString('\n')
		}
		if err != nil {
			errCh <- err
			return
		}

		lineCh <- strings.TrimRight(line, "\r\n")
	}()

	select {
	case err := <-errCh:
		return "", err
	case line := <-lineCh:
		return line, nil
	case <-sigCh:
		// Print a newline so that any further output starts properly
		// on a new line.
		fmt.Fprintln(u.Writer)

		return "", errors.New("interrupted")
	}
}

func (u *BasicUi) Error(message string) {
	w := u.Writer
	if u.ErrorWriter != nil {
		w = u.ErrorWriter
	}

	fmt.Fprint(w, message)
	fmt.Fprint(w, "\n")
}

func (u *BasicUi) Info(message string) {
	u.Output(message)
}

func (u *BasicUi) Output(message string) {
	fmt.Fprint(u.Writer, message)
	fmt.Fprint(u.Writer, "\n")
}

func (u *BasicUi) Warn(message string) {
	u.Error(message)
}

// PrefixedUi is an implementation of Ui that prefixes messages.
type PrefixedUi struct {
	AskPrefix       string
	AskSecretPrefix string
	OutputPrefix    string
	InfoPrefix      string
	ErrorPrefix     string
	WarnPrefix      string
	Ui              Ui
}

func (u *PrefixedUi) Ask(query string) (string, error) {
	if query != "" {
		query = fmt.Sprintf("%s%s", u.AskPrefix, query)
	}

	return u.Ui.Ask(query)
}

func (u *PrefixedUi) AskSecret(query string) (string, error) {
	if query != "" {
		query = fmt.Sprintf("%s%s", u.AskSecretPrefix, query)
	}

	return u.Ui.AskSecret(query)
}

func (u *PrefixedUi) Error(message string) {
	if message != "" {
		message = fmt.Sprintf("%s%s", u.ErrorPrefix, message)
	}

	u.Ui.Error(message)
}

func (u *PrefixedUi) Info(message string) {
	if message != "" {
		message = fmt.Sprintf("%s%s", u.InfoPrefix, message)
	}

	u.Ui.Info(message)
}

func (u *PrefixedUi) Output(message string) {
	if message != "" {
		message = fmt.Sprintf("%s%s", u.OutputPrefix, message)
	}

	u.Ui.Output(message)
}

func (u *PrefixedUi) Warn(message string) {
	if message != "" {
		message = fmt.Sprintf("%s%s", u.WarnPrefix, message)
	}

	u.Ui.Warn(message)
}
