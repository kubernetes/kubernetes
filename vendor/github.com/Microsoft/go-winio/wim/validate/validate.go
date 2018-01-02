package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Microsoft/go-winio/wim"
)

func main() {
	flag.Parse()
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		panic(err)
	}

	w, err := wim.NewReader(f)
	if err != nil {
		panic(err)

	}

	fmt.Printf("%#v\n%#v\n", w.Image[0], w.Image[0].Windows)

	dir, err := w.Image[0].Open()
	if err != nil {
		panic(err)
	}

	err = recur(dir)
	if err != nil {
		panic(err)
	}
}

func recur(d *wim.File) error {
	files, err := d.Readdir()
	if err != nil {
		return fmt.Errorf("%s: %s", d.Name, err)
	}
	for _, f := range files {
		if f.IsDir() {
			err = recur(f)
			if err != nil {
				return fmt.Errorf("%s: %s", f.Name, err)
			}
		}
	}
	return nil
}
