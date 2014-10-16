// Command mc prints in multiple columns.
//
//   Usage: mc [-] [-N] [file...]
//
// Mc splits the input into as many columns as will fit in N
// print positions. If the output is a tty, the default N is
// the number of characters in a terminal line; otherwise the
// default N is 80. Under option - each input line ending in
// a colon ':' is printed separately.
package main

import (
	"github.com/kr/pty"
	"github.com/kr/text/colwriter"
	"io"
	"log"
	"os"
	"strconv"
)

func main() {
	var width int
	var flag uint
	args := os.Args[1:]
	for len(args) > 0 && len(args[0]) > 0 && args[0][0] == '-' {
		if len(args[0]) > 1 {
			width, _ = strconv.Atoi(args[0][1:])
		} else {
			flag |= colwriter.BreakOnColon
		}
		args = args[1:]
	}
	if width < 1 {
		_, width, _ = pty.Getsize(os.Stdout)
	}
	if width < 1 {
		width = 80
	}

	w := colwriter.NewWriter(os.Stdout, width, flag)
	if len(args) > 0 {
		for _, s := range args {
			if f, err := os.Open(s); err == nil {
				copyin(w, f)
				f.Close()
			} else {
				log.Println(err)
			}
		}
	} else {
		copyin(w, os.Stdin)
	}
}

func copyin(w *colwriter.Writer, r io.Reader) {
	if _, err := io.Copy(w, r); err != nil {
		log.Println(err)
	}
	if err := w.Flush(); err != nil {
		log.Println(err)
	}
}
