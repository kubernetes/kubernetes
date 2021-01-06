// This work is subject to the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
// license. Its contents can be found at:
// http://creativecommons.org/publicdomain/zero/1.0/

package bindata

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"unicode"
	"unicode/utf8"
)

// writeReleaseFunctions writes the release code file.
func writeReleaseFunctions(w io.Writer, c *Config, toc []Asset) error {
	err := writeReleaseHeader(w, c)
	if err != nil {
		return err
	}

	for i := range toc {
		if err := writeReleaseAsset(w, c, &toc[i]); err != nil {
			return err
		}
	}

	return nil
}

// writeReleaseHeader writes output file headers.
// This targets release builds.
func writeReleaseHeader(w io.Writer, c *Config) error {
	var err error
	if c.NoCompress {
		if c.NoMemCopy {
			err = header_uncompressed_nomemcopy(w)
		} else {
			err = header_uncompressed_memcopy(w)
		}
	} else {
		if c.NoMemCopy {
			err = header_compressed_nomemcopy(w)
		} else {
			err = header_compressed_memcopy(w)
		}
	}
	if err != nil {
		return err
	}
	return header_release_common(w)
}

// writeReleaseAsset write a release entry for the given asset.
// A release entry is a function which embeds and returns
// the file's byte content.
func writeReleaseAsset(w io.Writer, c *Config, asset *Asset) error {
	fd, err := os.Open(asset.Path)
	if err != nil {
		return err
	}

	defer fd.Close()

	h := sha256.New()
	tr := io.TeeReader(fd, h)
	if c.NoCompress {
		if c.NoMemCopy {
			err = uncompressed_nomemcopy(w, asset, tr)
		} else {
			err = uncompressed_memcopy(w, asset, tr)
		}
	} else {
		if c.NoMemCopy {
			err = compressed_nomemcopy(w, asset, tr)
		} else {
			err = compressed_memcopy(w, asset, tr)
		}
	}
	if err != nil {
		return err
	}
	var digest [sha256.Size]byte
	copy(digest[:], h.Sum(nil))
	return asset_release_common(w, c, asset, digest)
}

var (
	backquote = []byte("`")
	bom       = []byte("\xEF\xBB\xBF")
)

// sanitize prepares a valid UTF-8 string as a raw string constant.
// Based on https://code.google.com/p/go/source/browse/godoc/static/makestatic.go?repo=tools
func sanitize(b []byte) []byte {
	var chunks [][]byte
	for i, b := range bytes.Split(b, backquote) {
		if i > 0 {
			chunks = append(chunks, backquote)
		}
		for j, c := range bytes.Split(b, bom) {
			if j > 0 {
				chunks = append(chunks, bom)
			}
			if len(c) > 0 {
				chunks = append(chunks, c)
			}
		}
	}

	var buf bytes.Buffer
	sanitizeChunks(&buf, chunks)
	return buf.Bytes()
}

func sanitizeChunks(buf *bytes.Buffer, chunks [][]byte) {
	n := len(chunks)
	if n >= 2 {
		buf.WriteString("(")
		sanitizeChunks(buf, chunks[:n/2])
		buf.WriteString(" + ")
		sanitizeChunks(buf, chunks[n/2:])
		buf.WriteString(")")
		return
	}
	b := chunks[0]
	if bytes.Equal(b, backquote) {
		buf.WriteString("\"`\"")
		return
	}
	if bytes.Equal(b, bom) {
		buf.WriteString(`"\xEF\xBB\xBF"`)
		return
	}
	buf.WriteString("`")
	buf.Write(b)
	buf.WriteString("`")
}

func header_compressed_nomemcopy(w io.Writer) error {
	_, err := fmt.Fprintf(w, `import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func bindataRead(data, name string) ([]byte, error) {
	gz, err := gzip.NewReader(strings.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("read %%q: %`+wrappedError+`", name, err)
	}

	var buf bytes.Buffer
	_, err = io.Copy(&buf, gz)

	if err != nil {
		return nil, fmt.Errorf("read %%q: %`+wrappedError+`", name, err)
	}

	clErr := gz.Close()
	if clErr != nil {
		return nil, clErr
	}

	return buf.Bytes(), nil
}

`)
	return err
}

func header_compressed_memcopy(w io.Writer) error {
	_, err := fmt.Fprintf(w, `import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func bindataRead(data []byte, name string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("read %%q: %`+wrappedError+`", name, err)
	}

	var buf bytes.Buffer
	_, err = io.Copy(&buf, gz)
	clErr := gz.Close()

	if err != nil {
		return nil, fmt.Errorf("read %%q: %`+wrappedError+`", name, err)
	}
	if clErr != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

`)
	return err
}

func header_uncompressed_nomemcopy(w io.Writer) error {
	_, err := fmt.Fprintf(w, `import (
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"
	"unsafe"
)

func bindataRead(data, name string) ([]byte, error) {
	var empty [0]byte
	sx := (*reflect.StringHeader)(unsafe.Pointer(&data))
	b := empty[:]
	bx := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	bx.Data = sx.Data
	bx.Len = len(data)
	bx.Cap = bx.Len
	return b, nil
}

`)
	return err
}

func header_uncompressed_memcopy(w io.Writer) error {
	_, err := fmt.Fprintf(w, `import (
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

`)
	return err
}

func header_release_common(w io.Writer) error {
	_, err := fmt.Fprintf(w, `type asset struct {
	bytes  []byte
	info   os.FileInfo
	digest [sha256.Size]byte
}

type bindataFileInfo struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
}

func (fi bindataFileInfo) Name() string {
	return fi.name
}
func (fi bindataFileInfo) Size() int64 {
	return fi.size
}
func (fi bindataFileInfo) Mode() os.FileMode {
	return fi.mode
}
func (fi bindataFileInfo) ModTime() time.Time {
	return fi.modTime
}
func (fi bindataFileInfo) IsDir() bool {
	return false
}
func (fi bindataFileInfo) Sys() interface{} {
	return nil
}

`)
	return err
}

func compressed_nomemcopy(w io.Writer, asset *Asset, r io.Reader) error {
	_, err := fmt.Fprintf(w, `var _%s = "`, asset.Func)
	if err != nil {
		return err
	}

	gz := gzip.NewWriter(&StringWriter{Writer: w})
	_, err = io.Copy(gz, r)
	gz.Close()

	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, `"

func %sBytes() ([]byte, error) {
	return bindataRead(
		_%s,
		%q,
	)
}

`, asset.Func, asset.Func, asset.Name)
	return err
}

func compressed_memcopy(w io.Writer, asset *Asset, r io.Reader) error {
	_, err := fmt.Fprintf(w, `var _%s = []byte("`, asset.Func)
	if err != nil {
		return err
	}

	gz := gzip.NewWriter(&StringWriter{Writer: w})
	_, err = io.Copy(gz, r)
	gz.Close()

	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, `")

func %sBytes() ([]byte, error) {
	return bindataRead(
		_%s,
		%q,
	)
}

`, asset.Func, asset.Func, asset.Name)
	return err
}

func uncompressed_nomemcopy(w io.Writer, asset *Asset, r io.Reader) error {
	_, err := fmt.Fprintf(w, `var _%s = "`, asset.Func)
	if err != nil {
		return err
	}

	_, err = io.Copy(&StringWriter{Writer: w}, r)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, `"

func %sBytes() ([]byte, error) {
	return bindataRead(
		_%s,
		%q,
	)
}

`, asset.Func, asset.Func, asset.Name)
	return err
}

func validSanitizedUtf8(b []byte) bool {
	if !utf8.Valid(b) {
		return false
	}
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		if r == 0 {
			return false
		}
		if unicode.In(r, unicode.Cf) {
			// staticcheck doesn't like these; fallback to slow decoder
			return false
		}
		b = b[size:]
	}
	return true
}

func uncompressed_memcopy(w io.Writer, asset *Asset, r io.Reader) error {
	_, err := fmt.Fprintf(w, `var _%s = []byte(`, asset.Func)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}
	if len(b) > 0 && validSanitizedUtf8(b) {
		w.Write(sanitize(b))
	} else {
		fmt.Fprintf(w, "%+q", b)
	}

	_, err = fmt.Fprintf(w, `)

func %sBytes() ([]byte, error) {
	return _%s, nil
}

`, asset.Func, asset.Func)
	return err
}

func asset_release_common(w io.Writer, c *Config, asset *Asset, digest [sha256.Size]byte) error {
	fi, err := os.Stat(asset.Path)
	if err != nil {
		return err
	}

	mode := uint(fi.Mode())
	modTime := fi.ModTime().Unix()
	size := fi.Size()
	if c.NoMetadata {
		mode = 0
		modTime = 0
		size = 0
	}
	if c.Mode > 0 {
		mode = uint(os.ModePerm) & c.Mode
	}
	if c.ModTime > 0 {
		modTime = c.ModTime
	}
	_, err = fmt.Fprintf(w, `func %s() (*asset, error) {
	bytes, err := %sBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: %q, size: %d, mode: os.FileMode(%#o), modTime: time.Unix(%d, 0)}
	a := &asset{bytes: bytes, info: info, digest: %#v}
	return a, nil
}

`, asset.Func, asset.Func, asset.Name, size, mode, modTime, digest)
	return err
}
