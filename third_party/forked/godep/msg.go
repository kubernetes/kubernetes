package main

import (
	"fmt"
	"log"

	"github.com/kr/pretty"
)

func debugln(a ...interface{}) (int, error) {
	if debug {
		return fmt.Println(a...)
	}
	return 0, nil
}

func verboseln(a ...interface{}) {
	if verbose {
		log.Println(a...)
	}
}

func debugf(format string, a ...interface{}) (int, error) {
	if debug {
		return fmt.Printf(format, a...)
	}
	return 0, nil
}

func ppln(a ...interface{}) (int, error) {
	if debug {
		return pretty.Println(a...)
	}
	return 0, nil
}
