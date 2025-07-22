package internal

import (
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

func Debug(name string, mask int32) {
	names := []struct {
		n string
		m int32
	}{
		{"FILE_ACCESS", unix.FILE_ACCESS},
		{"FILE_MODIFIED", unix.FILE_MODIFIED},
		{"FILE_ATTRIB", unix.FILE_ATTRIB},
		{"FILE_TRUNC", unix.FILE_TRUNC},
		{"FILE_NOFOLLOW", unix.FILE_NOFOLLOW},
		{"FILE_DELETE", unix.FILE_DELETE},
		{"FILE_RENAME_TO", unix.FILE_RENAME_TO},
		{"FILE_RENAME_FROM", unix.FILE_RENAME_FROM},
		{"UNMOUNTED", unix.UNMOUNTED},
		{"MOUNTEDOVER", unix.MOUNTEDOVER},
		{"FILE_EXCEPTION", unix.FILE_EXCEPTION},
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
	fmt.Fprintf(os.Stderr, "FSNOTIFY_DEBUG: %s  %10d:%-30s → %q\n",
		time.Now().Format("15:04:05.000000000"), mask, strings.Join(l, " | "), name)
}
