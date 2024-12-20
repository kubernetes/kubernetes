package internal

import "golang.org/x/sys/unix"

var names = []struct {
	n string
	m uint32
}{
	{"NOTE_ATTRIB", unix.NOTE_ATTRIB},
	{"NOTE_CHILD", unix.NOTE_CHILD},
	{"NOTE_DELETE", unix.NOTE_DELETE},
	{"NOTE_EXEC", unix.NOTE_EXEC},
	{"NOTE_EXIT", unix.NOTE_EXIT},
	{"NOTE_EXTEND", unix.NOTE_EXTEND},
	{"NOTE_FORK", unix.NOTE_FORK},
	{"NOTE_LINK", unix.NOTE_LINK},
	{"NOTE_LOWAT", unix.NOTE_LOWAT},
	{"NOTE_PCTRLMASK", unix.NOTE_PCTRLMASK},
	{"NOTE_PDATAMASK", unix.NOTE_PDATAMASK},
	{"NOTE_RENAME", unix.NOTE_RENAME},
	{"NOTE_REVOKE", unix.NOTE_REVOKE},
	{"NOTE_TRACK", unix.NOTE_TRACK},
	{"NOTE_TRACKERR", unix.NOTE_TRACKERR},
	{"NOTE_WRITE", unix.NOTE_WRITE},
}
