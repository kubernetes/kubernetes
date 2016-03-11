// Package browser provides helpers to open files, readers, and urls in a browser window.
//
// The choice of which browser is started is entirely client dependant.
package browser

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
)

// Stdout is the io.Writer to which executed commands write standard output.
var Stdout io.Writer = os.Stdout

// Stderr is the io.Writer to which executed commands write standard error.
var Stderr io.Writer = os.Stderr

// OpenFile opens new browser window for the file path.
func OpenFile(path string) error {
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	return OpenURL("file://" + path)
}

// OpenReader consumes the contents of r and presents the
// results in a new browser window.
func OpenReader(r io.Reader) error {
	f, err := ioutil.TempFile("", "browser")
	if err != nil {
		return fmt.Errorf("browser: could not create temporary file: %v", err)
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return fmt.Errorf("browser: caching temporary file failed: %v", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("browser: caching temporary file failed: %v", err)
	}
	oldname := f.Name()
	newname := oldname + ".html"
	if err := os.Rename(oldname, newname); err != nil {
		return fmt.Errorf("browser: renaming temporary file failed: %v", err)
	}
	return OpenFile(newname)
}

// OpenURL opens a new browser window pointing to url.
func OpenURL(url string) error {
	return openBrowser(url)
}

func runCmd(prog string, args ...string) error {
	cmd := exec.Command(prog, args...)
	cmd.Stdout = Stdout
	cmd.Stderr = Stderr
	return cmd.Run()
}
