package main

import (
	"fmt"
	"io"
	"os"

	"github.com/cespare/xxhash/v2"
)

func main() {
	if contains(os.Args[1:], "-h") {
		fmt.Fprintf(os.Stderr, `Usage:
  %s [filenames]
If no filenames are provided or only - is given, input is read from stdin.
`, os.Args[0])
		os.Exit(1)
	}
	if len(os.Args) < 2 || len(os.Args) == 2 && os.Args[1] == "-" {
		printHash(os.Stdin, "-")
		return
	}
	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		printHash(f, path)
		f.Close()
	}
}

func contains(ss []string, s string) bool {
	for _, s1 := range ss {
		if s1 == s {
			return true
		}
	}
	return false
}

func printHash(r io.Reader, name string) {
	h := xxhash.New()
	if _, err := io.Copy(h, r); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	fmt.Printf("%016x  %s\n", h.Sum64(), name)
}
