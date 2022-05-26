// +build !windows

package dotwriter

import (
	"fmt"
	"strings"
)

// clear the line and move the cursor up
var clear = fmt.Sprintf("%c[%dA%c[2K", ESC, 1, ESC)

func (w *Writer) clearLines(count int) {
	_, _ = fmt.Fprint(w.out, strings.Repeat(clear, count))
}
