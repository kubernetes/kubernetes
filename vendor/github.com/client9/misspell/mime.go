package misspell

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// The number of possible binary formats is very large
// items that might be checked into a repo or be an
// artifact of a build.  Additions welcome.
//
// Golang's internal table is very small and can't be
// relied on.  Even then things like ".js" have a mime
// type of "application/javascipt" which isn't very helpful.
// "[x]" means we have  sniff test and suffix test should be eliminated
var binary = map[string]bool{
	".a":     true, // [ ] archive
	".bin":   true, // [ ] binary
	".bz2":   true, // [ ] compression
	".class": true, // [x] Java class file
	".dll":   true, // [ ] shared library
	".exe":   true, // [ ] binary
	".gif":   true, // [ ] image
	".gpg":   true, // [x] text, but really all base64
	".gz":    true, // [ ] compression
	".ico":   true, // [ ] image
	".jar":   true, // [x] archive
	".jpeg":  true, // [ ] image
	".jpg":   true, // [ ] image
	".mp3":   true, // [ ] audio
	".mp4":   true, // [ ] video
	".mpeg":  true, // [ ] video
	".o":     true, // [ ] object file
	".pdf":   true, // [x] pdf
	".png":   true, // [x] image
	".pyc":   true, // [ ] Python bytecode
	".pyo":   true, // [ ] Python bytecode
	".so":    true, // [x] shared library
	".swp":   true, // [ ] vim swap file
	".tar":   true, // [ ] archive
	".tiff":  true, // [ ] image
	".woff":  true, // [ ] font
	".woff2": true, // [ ] font
	".xz":    true, // [ ] compression
	".z":     true, // [ ] compression
	".zip":   true, // [x] archive
}

// isBinaryFilename returns true if the file is likely to be binary
//
// Better heuristics could be done here, in particular a binary
// file is unlikely to be UTF-8 encoded.  However this is cheap
// and will solve the immediate need of making sure common
// binary formats are not corrupted by mistake.
func isBinaryFilename(s string) bool {
	return binary[strings.ToLower(filepath.Ext(s))]
}

var scm = map[string]bool{
	".bzr": true,
	".git": true,
	".hg":  true,
	".svn": true,
	"CVS":  true,
}

// isSCMPath returns true if the path is likely part of a (private) SCM
//  directory.  E.g.  ./git/something  = true
func isSCMPath(s string) bool {
	// hack for .git/COMMIT_EDITMSG and .git/TAG_EDITMSG
	// normally we don't look at anything in .git
	// but COMMIT_EDITMSG and TAG_EDITMSG are used as
	// temp files for git commits.  Allowing misspell to inspect
	// these files allows for commit-msg hooks
	// https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
	if strings.Contains(filepath.Base(s), "EDITMSG") {
		return false
	}
	parts := strings.Split(filepath.Clean(s), string(filepath.Separator))
	for _, dir := range parts {
		if scm[dir] {
			return true
		}
	}
	return false
}

var magicHeaders = [][]byte{
	// Issue #68
	// PGP messages and signatures are "text" but really just
	// blobs of base64-text and should not be misspell-checked
	[]byte("-----BEGIN PGP MESSAGE-----"),
	[]byte("-----BEGIN PGP SIGNATURE-----"),

	// ELF
	{0x7f, 0x45, 0x4c, 0x46},

	// Postscript
	{0x25, 0x21, 0x50, 0x53},

	// PDF
	{0x25, 0x50, 0x44, 0x46},

	// Java class file
	// https://en.wikipedia.org/wiki/Java_class_file
	{0xCA, 0xFE, 0xBA, 0xBE},

	// PNG
	// https://en.wikipedia.org/wiki/Portable_Network_Graphics
	{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a},

	// ZIP, JAR, ODF, OOXML
	{0x50, 0x4B, 0x03, 0x04},
	{0x50, 0x4B, 0x05, 0x06},
	{0x50, 0x4B, 0x07, 0x08},
}

func isTextFile(raw []byte) bool {
	for _, magic := range magicHeaders {
		if bytes.HasPrefix(raw, magic) {
			return false
		}
	}

	// allow any text/ type with utf-8 encoding
	// DetectContentType sometimes returns charset=utf-16 for XML stuff
	//  in which case ignore.
	mime := http.DetectContentType(raw)
	return strings.HasPrefix(mime, "text/") && strings.HasSuffix(mime, "charset=utf-8")
}

// ReadTextFile returns the contents of a file, first testing if it is a text file
//  returns ("", nil) if not a text file
//  returns ("", error) if error
//  returns (string, nil) if text
//
// unfortunately, in worse case, this does
//   1 stat
//   1 open,read,close of 512 bytes
//   1 more stat,open, read everything, close (via ioutil.ReadAll)
//  This could be kinder to the filesystem.
//
// This uses some heuristics of the file's extension (e.g. .zip, .txt) and
// uses a sniffer to determine if the file is text or not.
// Using file extensions isn't great, but probably
// good enough for real-world use.
// Golang's built in sniffer is problematic for differnet reasons.  It's
// optimized for HTML, and is very limited in detection.  It would be good
// to explicitly add some tests for ELF/DWARF formats to make sure we never
// corrupt binary files.
func ReadTextFile(filename string) (string, error) {
	if isBinaryFilename(filename) {
		return "", nil
	}

	if isSCMPath(filename) {
		return "", nil
	}

	fstat, err := os.Stat(filename)

	if err != nil {
		return "", fmt.Errorf("Unable to stat %q: %s", filename, err)
	}

	// directory: nothing to do.
	if fstat.IsDir() {
		return "", nil
	}

	// avoid reading in multi-gig files
	// if input is large, read the first 512 bytes to sniff type
	// if not-text, then exit
	isText := false
	if fstat.Size() > 50000 {
		fin, err := os.Open(filename)
		if err != nil {
			return "", fmt.Errorf("Unable to open large file %q: %s", filename, err)
		}
		defer fin.Close()
		buf := make([]byte, 512)
		_, err = io.ReadFull(fin, buf)
		if err != nil {
			return "", fmt.Errorf("Unable to read 512 bytes from %q: %s", filename, err)
		}
		if !isTextFile(buf) {
			return "", nil
		}

		// set so we don't double check this file
		isText = true
	}

	// read in whole file
	raw, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("Unable to read all %q: %s", filename, err)
	}

	if !isText && !isTextFile(raw) {
		return "", nil
	}
	return string(raw), nil
}
