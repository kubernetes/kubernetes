// Open is a simple example of the github.com/pkg/browser package.
//
// Usage:
//
//    # Open a file in a browser window
//    Open $FILE
//
//    # Open a URL in a browser window
//    Open $URL
//
//    # Open the contents of stdin in a browser window
//    cat $SOMEFILE | Open
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/pkg/browser"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage:\n  %s [file]\n", os.Args[0])
	flag.PrintDefaults()
}

func init() {
	flag.Usage = usage
	flag.Parse()
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	args := flag.Args()
	switch len(args) {
	case 0:
		check(browser.OpenReader(os.Stdin))
	case 1:
		check(browser.OpenFile(args[0]))
	default:
		usage()
	}
}
