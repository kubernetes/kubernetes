package fs_test

import (
	"fmt"
	"os"

	"github.com/kr/fs"
)

func ExampleWalker() {
	walker := fs.Walk("/usr/lib")
	for walker.Step() {
		if err := walker.Err(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		fmt.Println(walker.Path())
	}
}
