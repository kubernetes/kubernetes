package ini

import (
	"reflect"
	"syscall"
	"testing"
)

func TestLoadFile(t *testing.T) {
	originalOpenFiles := numFilesOpen(t)

	file, err := LoadFile("test.ini")
	if err != nil {
		t.Fatal(err)
	}

	if originalOpenFiles != numFilesOpen(t) {
		t.Error("test.ini not closed")
	}

	if !reflect.DeepEqual(file, File{"default": {"stuff": "things"}}) {
		t.Error("file not read correctly")
	}
}

func numFilesOpen(t *testing.T) (num uint64) {
	var rlimit syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rlimit)
	if err != nil {
		t.Fatal(err)
	}
	maxFds := int(rlimit.Cur)

	var stat syscall.Stat_t
	for i := 0; i < maxFds; i++ {
		if syscall.Fstat(i, &stat) == nil {
			num++
		} else {
			return
		}
	}
	return
}
