package internal

import (
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

func Debug(name string, mask, cookie uint32) {
	names := []struct {
		n string
		m uint32
	}{
		{"IN_ACCESS", unix.IN_ACCESS},
		{"IN_ATTRIB", unix.IN_ATTRIB},
		{"IN_CLOSE", unix.IN_CLOSE},
		{"IN_CLOSE_NOWRITE", unix.IN_CLOSE_NOWRITE},
		{"IN_CLOSE_WRITE", unix.IN_CLOSE_WRITE},
		{"IN_CREATE", unix.IN_CREATE},
		{"IN_DELETE", unix.IN_DELETE},
		{"IN_DELETE_SELF", unix.IN_DELETE_SELF},
		{"IN_IGNORED", unix.IN_IGNORED},
		{"IN_ISDIR", unix.IN_ISDIR},
		{"IN_MODIFY", unix.IN_MODIFY},
		{"IN_MOVE", unix.IN_MOVE},
		{"IN_MOVED_FROM", unix.IN_MOVED_FROM},
		{"IN_MOVED_TO", unix.IN_MOVED_TO},
		{"IN_MOVE_SELF", unix.IN_MOVE_SELF},
		{"IN_OPEN", unix.IN_OPEN},
		{"IN_Q_OVERFLOW", unix.IN_Q_OVERFLOW},
		{"IN_UNMOUNT", unix.IN_UNMOUNT},
	}

	var (
		l       []string
		unknown = mask
	)
	for _, n := range names {
		if mask&n.m == n.m {
			l = append(l, n.n)
			unknown ^= n.m
		}
	}
	if unknown > 0 {
		l = append(l, fmt.Sprintf("0x%x", unknown))
	}
	var c string
	if cookie > 0 {
		c = fmt.Sprintf("(cookie: %d) ", cookie)
	}
	fmt.Fprintf(os.Stderr, "FSNOTIFY_DEBUG: %s  %-30s → %s%q\n",
		time.Now().Format("15:04:05.000000000"), strings.Join(l, "|"), c, name)
}
