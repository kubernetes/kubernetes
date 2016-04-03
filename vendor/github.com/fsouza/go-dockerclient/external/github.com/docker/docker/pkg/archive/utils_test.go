package archive

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"
)

var testUntarFns = map[string]func(string, io.Reader) error{
	"untar": func(dest string, r io.Reader) error {
		return Untar(r, dest, nil)
	},
	"applylayer": func(dest string, r io.Reader) error {
		_, err := ApplyLayer(dest, Reader(r))
		return err
	},
}

// testBreakout is a helper function that, within the provided `tmpdir` directory,
// creates a `victim` folder with a generated `hello` file in it.
// `untar` extracts to a directory named `dest`, the tar file created from `headers`.
//
// Here are the tested scenarios:
// - removed `victim` folder				(write)
// - removed files from `victim` folder			(write)
// - new files in `victim` folder			(write)
// - modified files in `victim` folder			(write)
// - file in `dest` with same content as `victim/hello` (read)
//
// When using testBreakout make sure you cover one of the scenarios listed above.
func testBreakout(untarFn string, tmpdir string, headers []*tar.Header) error {
	tmpdir, err := ioutil.TempDir("", tmpdir)
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpdir)

	dest := filepath.Join(tmpdir, "dest")
	if err := os.Mkdir(dest, 0755); err != nil {
		return err
	}

	victim := filepath.Join(tmpdir, "victim")
	if err := os.Mkdir(victim, 0755); err != nil {
		return err
	}
	hello := filepath.Join(victim, "hello")
	helloData, err := time.Now().MarshalText()
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(hello, helloData, 0644); err != nil {
		return err
	}
	helloStat, err := os.Stat(hello)
	if err != nil {
		return err
	}

	reader, writer := io.Pipe()
	go func() {
		t := tar.NewWriter(writer)
		for _, hdr := range headers {
			t.WriteHeader(hdr)
		}
		t.Close()
	}()

	untar := testUntarFns[untarFn]
	if untar == nil {
		return fmt.Errorf("could not find untar function %q in testUntarFns", untarFn)
	}
	if err := untar(dest, reader); err != nil {
		if _, ok := err.(breakoutError); !ok {
			// If untar returns an error unrelated to an archive breakout,
			// then consider this an unexpected error and abort.
			return err
		}
		// Here, untar detected the breakout.
		// Let's move on verifying that indeed there was no breakout.
		fmt.Printf("breakoutError: %v\n", err)
	}

	// Check victim folder
	f, err := os.Open(victim)
	if err != nil {
		// codepath taken if victim folder was removed
		return fmt.Errorf("archive breakout: error reading %q: %v", victim, err)
	}
	defer f.Close()

	// Check contents of victim folder
	//
	// We are only interested in getting 2 files from the victim folder, because if all is well
	// we expect only one result, the `hello` file. If there is a second result, it cannot
	// hold the same name `hello` and we assume that a new file got created in the victim folder.
	// That is enough to detect an archive breakout.
	names, err := f.Readdirnames(2)
	if err != nil {
		// codepath taken if victim is not a folder
		return fmt.Errorf("archive breakout: error reading directory content of %q: %v", victim, err)
	}
	for _, name := range names {
		if name != "hello" {
			// codepath taken if new file was created in victim folder
			return fmt.Errorf("archive breakout: new file %q", name)
		}
	}

	// Check victim/hello
	f, err = os.Open(hello)
	if err != nil {
		// codepath taken if read permissions were removed
		return fmt.Errorf("archive breakout: could not lstat %q: %v", hello, err)
	}
	defer f.Close()
	b, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	if helloStat.IsDir() != fi.IsDir() ||
		// TODO: cannot check for fi.ModTime() change
		helloStat.Mode() != fi.Mode() ||
		helloStat.Size() != fi.Size() ||
		!bytes.Equal(helloData, b) {
		// codepath taken if hello has been modified
		return fmt.Errorf("archive breakout: file %q has been modified. Contents: expected=%q, got=%q. FileInfo: expected=%#v, got=%#v", hello, helloData, b, helloStat, fi)
	}

	// Check that nothing in dest/ has the same content as victim/hello.
	// Since victim/hello was generated with time.Now(), it is safe to assume
	// that any file whose content matches exactly victim/hello, managed somehow
	// to access victim/hello.
	return filepath.Walk(dest, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			if err != nil {
				// skip directory if error
				return filepath.SkipDir
			}
			// enter directory
			return nil
		}
		if err != nil {
			// skip file if error
			return nil
		}
		b, err := ioutil.ReadFile(path)
		if err != nil {
			// Houston, we have a problem. Aborting (space)walk.
			return err
		}
		if bytes.Equal(helloData, b) {
			return fmt.Errorf("archive breakout: file %q has been accessed via %q", hello, path)
		}
		return nil
	})
}
