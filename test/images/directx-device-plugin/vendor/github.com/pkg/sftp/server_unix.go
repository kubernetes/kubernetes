// +build darwin dragonfly freebsd !android,linux netbsd openbsd solaris
// +build cgo

package sftp

import (
	"fmt"
	"os"
	"path"
	"syscall"
	"time"
)

func runLsTypeWord(dirent os.FileInfo) string {
	// find first character, the type char
	// b     Block special file.
	// c     Character special file.
	// d     Directory.
	// l     Symbolic link.
	// s     Socket link.
	// p     FIFO.
	// -     Regular file.
	tc := '-'
	mode := dirent.Mode()
	if (mode & os.ModeDir) != 0 {
		tc = 'd'
	} else if (mode & os.ModeDevice) != 0 {
		tc = 'b'
		if (mode & os.ModeCharDevice) != 0 {
			tc = 'c'
		}
	} else if (mode & os.ModeSymlink) != 0 {
		tc = 'l'
	} else if (mode & os.ModeSocket) != 0 {
		tc = 's'
	} else if (mode & os.ModeNamedPipe) != 0 {
		tc = 'p'
	}

	// owner
	orc := '-'
	if (mode & 0400) != 0 {
		orc = 'r'
	}
	owc := '-'
	if (mode & 0200) != 0 {
		owc = 'w'
	}
	oxc := '-'
	ox := (mode & 0100) != 0
	setuid := (mode & os.ModeSetuid) != 0
	if ox && setuid {
		oxc = 's'
	} else if setuid {
		oxc = 'S'
	} else if ox {
		oxc = 'x'
	}

	// group
	grc := '-'
	if (mode & 040) != 0 {
		grc = 'r'
	}
	gwc := '-'
	if (mode & 020) != 0 {
		gwc = 'w'
	}
	gxc := '-'
	gx := (mode & 010) != 0
	setgid := (mode & os.ModeSetgid) != 0
	if gx && setgid {
		gxc = 's'
	} else if setgid {
		gxc = 'S'
	} else if gx {
		gxc = 'x'
	}

	// all / others
	arc := '-'
	if (mode & 04) != 0 {
		arc = 'r'
	}
	awc := '-'
	if (mode & 02) != 0 {
		awc = 'w'
	}
	axc := '-'
	ax := (mode & 01) != 0
	sticky := (mode & os.ModeSticky) != 0
	if ax && sticky {
		axc = 't'
	} else if sticky {
		axc = 'T'
	} else if ax {
		axc = 'x'
	}

	return fmt.Sprintf("%c%c%c%c%c%c%c%c%c%c", tc, orc, owc, oxc, grc, gwc, gxc, arc, awc, axc)
}

func runLsStatt(dirname string, dirent os.FileInfo, statt *syscall.Stat_t) string {
	// example from openssh sftp server:
	// crw-rw-rw-    1 root     wheel           0 Jul 31 20:52 ttyvd
	// format:
	// {directory / char device / etc}{rwxrwxrwx}  {number of links} owner group size month day [time (this year) | year (otherwise)] name

	typeword := runLsTypeWord(dirent)
	numLinks := statt.Nlink
	uid := statt.Uid
	gid := statt.Gid
	username := fmt.Sprintf("%d", uid)
	groupname := fmt.Sprintf("%d", gid)
	// TODO FIXME: uid -> username, gid -> groupname lookup for ls -l format output

	mtime := dirent.ModTime()
	monthStr := mtime.Month().String()[0:3]
	day := mtime.Day()
	year := mtime.Year()
	now := time.Now()
	isOld := mtime.Before(now.Add(-time.Hour * 24 * 365 / 2))

	yearOrTime := fmt.Sprintf("%02d:%02d", mtime.Hour(), mtime.Minute())
	if isOld {
		yearOrTime = fmt.Sprintf("%d", year)
	}

	return fmt.Sprintf("%s %4d %-8s %-8s %8d %s %2d %5s %s", typeword, numLinks, username, groupname, dirent.Size(), monthStr, day, yearOrTime, dirent.Name())
}

// ls -l style output for a file, which is in the 'long output' section of a readdir response packet
// this is a very simple (lazy) implementation, just enough to look almost like openssh in a few basic cases
func runLs(dirname string, dirent os.FileInfo) string {
	dsys := dirent.Sys()
	if dsys == nil {
	} else if statt, ok := dsys.(*syscall.Stat_t); !ok {
	} else {
		return runLsStatt(dirname, dirent, statt)
	}

	return path.Join(dirname, dirent.Name())
}
