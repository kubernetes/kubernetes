package tsm1

import (
	"fmt"
	"os"
)

func mmap(f *os.File, offset int64, length int) ([]byte, error) {
	return nil, fmt.Errorf("mmap file not supported windows")
}

func munmap(b []byte) (err error) {
	return nil, fmt.Errorf("munmap file not supported on windows")
}
