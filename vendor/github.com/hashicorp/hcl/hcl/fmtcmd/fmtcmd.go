// Derivative work from:
//	- https://golang.org/src/cmd/gofmt/gofmt.go
//	- https://github.com/fatih/hclfmt

package fmtcmd

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/hashicorp/hcl/hcl/printer"
)

var (
	ErrWriteStdin = errors.New("cannot use write option with standard input")
)

type Options struct {
	List  bool // list files whose formatting differs
	Write bool // write result to (source) file instead of stdout
	Diff  bool // display diffs of formatting changes
}

func isValidFile(f os.FileInfo, extensions []string) bool {
	if !f.IsDir() && !strings.HasPrefix(f.Name(), ".") {
		for _, ext := range extensions {
			if strings.HasSuffix(f.Name(), "."+ext) {
				return true
			}
		}
	}

	return false
}

// If in == nil, the source is the contents of the file with the given filename.
func processFile(filename string, in io.Reader, out io.Writer, stdin bool, opts Options) error {
	if in == nil {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer f.Close()
		in = f
	}

	src, err := ioutil.ReadAll(in)
	if err != nil {
		return err
	}

	res, err := printer.Format(src)
	if err != nil {
		return err
	}

	if !bytes.Equal(src, res) {
		// formatting has changed
		if opts.List {
			fmt.Fprintln(out, filename)
		}
		if opts.Write {
			err = ioutil.WriteFile(filename, res, 0644)
			if err != nil {
				return err
			}
		}
		if opts.Diff {
			data, err := diff(src, res)
			if err != nil {
				return fmt.Errorf("computing diff: %s", err)
			}
			fmt.Fprintf(out, "diff a/%s b/%s\n", filename, filename)
			out.Write(data)
		}
	}

	if !opts.List && !opts.Write && !opts.Diff {
		_, err = out.Write(res)
	}

	return err
}

func walkDir(path string, extensions []string, stdout io.Writer, opts Options) error {
	visitFile := func(path string, f os.FileInfo, err error) error {
		if err == nil && isValidFile(f, extensions) {
			err = processFile(path, nil, stdout, false, opts)
		}
		return err
	}

	return filepath.Walk(path, visitFile)
}

func Run(
	paths, extensions []string,
	stdin io.Reader,
	stdout io.Writer,
	opts Options,
) error {
	if len(paths) == 0 {
		if opts.Write {
			return ErrWriteStdin
		}
		if err := processFile("<standard input>", stdin, stdout, true, opts); err != nil {
			return err
		}
		return nil
	}

	for _, path := range paths {
		switch dir, err := os.Stat(path); {
		case err != nil:
			return err
		case dir.IsDir():
			if err := walkDir(path, extensions, stdout, opts); err != nil {
				return err
			}
		default:
			if err := processFile(path, nil, stdout, false, opts); err != nil {
				return err
			}
		}
	}

	return nil
}

func diff(b1, b2 []byte) (data []byte, err error) {
	f1, err := ioutil.TempFile("", "")
	if err != nil {
		return
	}
	defer os.Remove(f1.Name())
	defer f1.Close()

	f2, err := ioutil.TempFile("", "")
	if err != nil {
		return
	}
	defer os.Remove(f2.Name())
	defer f2.Close()

	f1.Write(b1)
	f2.Write(b2)

	data, err = exec.Command("diff", "-u", f1.Name(), f2.Name()).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		err = nil
	}
	return
}
