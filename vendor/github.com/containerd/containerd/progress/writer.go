package progress

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"github.com/containerd/console"
)

// Writer buffers writes until flush, at which time the last screen is cleared
// and the current buffer contents are written. This is useful for
// implementing progress displays, such as those implemented in docker and
// git.
type Writer struct {
	buf   bytes.Buffer
	w     io.Writer
	lines int
}

// NewWriter returns a writer
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w: w,
	}
}

// Write the provided bytes
func (w *Writer) Write(p []byte) (n int, err error) {
	return w.buf.Write(p)
}

// Flush should be called when refreshing the current display.
func (w *Writer) Flush() error {
	if w.buf.Len() == 0 {
		return nil
	}

	if err := w.clear(); err != nil {
		return err
	}

	ws, err := console.Current().Size()
	if err != nil {
		return fmt.Errorf("failed to get terminal width: %v", err)
	}
	strlines := strings.Split(w.buf.String(), "\n")
	w.lines = -1
	for _, line := range strlines {
		w.lines += len(line)/int(ws.Width) + 1
	}

	if _, err := w.w.Write(w.buf.Bytes()); err != nil {
		return err
	}

	w.buf.Reset()
	return nil
}

// TODO(stevvooe): The following are system specific. Break these out if we
// decide to build this package further.

func (w *Writer) clear() error {
	for i := 0; i < w.lines; i++ {
		if _, err := fmt.Fprintf(w.w, "\x1b[1A\x1b[2K\r"); err != nil {
			return err
		}
	}

	return nil
}
