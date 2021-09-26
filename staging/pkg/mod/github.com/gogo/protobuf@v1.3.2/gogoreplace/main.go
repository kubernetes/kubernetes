package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
)

func main() {
	args := os.Args
	if len(args) != 4 {
		fmt.Println("gogoreplace wants three arguments")
		fmt.Println("	gogoreplace oldsubstring newsubstring filename")
		os.Exit(1)
	}
	data, err := ioutil.ReadFile(args[3])
	if err != nil {
		panic(err)
	}
	data = bytes.Replace(data, []byte(args[1]), []byte(args[2]), -1)
	if err := ioutil.WriteFile(args[3], data, 0666); err != nil {
		panic(err)
	}
}
