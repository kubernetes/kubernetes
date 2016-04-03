package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/cpuguy83/go-md2man/md2man"
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

	out := md2man.Render(doc)

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
