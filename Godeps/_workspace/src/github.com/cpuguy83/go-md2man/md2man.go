package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/russross/blackfriday"
)

var inFilePath = flag.String("in", "", "Path to file to be processed")
var outFilePath = flag.String("out", "", "Path to output processed file")

func main() {
	flag.Parse()

	inFile, err := os.Open(*inFilePath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer inFile.Close()

	doc, err := ioutil.ReadAll(inFile)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	renderer := RoffRenderer(0)
	extensions := 0
	extensions |= blackfriday.EXTENSION_NO_INTRA_EMPHASIS
	extensions |= blackfriday.EXTENSION_TABLES
	extensions |= blackfriday.EXTENSION_FENCED_CODE
	extensions |= blackfriday.EXTENSION_AUTOLINK
	extensions |= blackfriday.EXTENSION_SPACE_HEADERS
	extensions |= blackfriday.EXTENSION_FOOTNOTES
	extensions |= blackfriday.EXTENSION_TITLEBLOCK

	out := blackfriday.Markdown(doc, renderer, extensions)

	outFile, err := os.Create(*outFilePath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer outFile.Close()
	_, err = outFile.Write(out)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
