// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"regexp"

	"github.com/mvdan/xurls"
)

var (
	matching = flag.String("m", "", "")
	relaxed  = flag.Bool("r", false, "")
)

func init() {
	flag.Usage = func() {
		p := func(format string, a ...interface{}) {
			fmt.Fprintf(os.Stderr, format, a...)
		}
		p("Usage: xurls [-h] [files]\n\n")
		p("If no files are given, it reads from standard input.\n\n")
		p("   -m <regexp>   only match urls whose scheme matches a regexp\n")
		p("                    example: 'https?://|mailto:'\n")
		p("   -r            also match urls without a scheme (relaxed)\n")
	}
}

func scanPath(re *regexp.Regexp, path string) error {
	r := os.Stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		for _, match := range re.FindAllString(word, -1) {
			fmt.Println(match)
		}
	}
	return scanner.Err()
}

func main() {
	flag.Parse()
	if *relaxed && *matching != "" {
		errExit(fmt.Errorf("-r and -m at the same time don't make much sense"))
	}
	re := xurls.Strict
	if *relaxed {
		re = xurls.Relaxed
	} else if *matching != "" {
		var err error
		if re, err = xurls.StrictMatchingScheme(*matching); err != nil {
			errExit(err)
		}
	}
	args := flag.Args()
	if len(args) == 0 {
		args = []string{"-"}
	}
	for _, path := range args {
		if err := scanPath(re, path); err != nil {
			errExit(err)
		}
	}
}

func errExit(err error) {
	fmt.Fprintf(os.Stderr, "%v\n", err)
	os.Exit(1)
}
