package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/docker/distribution/version"
	"github.com/opencontainers/go-digest"
)

var (
	algorithm   = digest.Canonical
	showVersion bool
)

type job struct {
	name   string
	reader io.Reader
}

func init() {
	flag.Var(&algorithm, "a", "select the digest algorithm (shorthand)")
	flag.Var(&algorithm, "algorithm", "select the digest algorithm")
	flag.BoolVar(&showVersion, "version", false, "show the version and exit")

	log.SetFlags(0)
	log.SetPrefix(os.Args[0] + ": ")
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: %s [files...]\n", os.Args[0])
	fmt.Fprint(os.Stderr, `
Calculate the digest of one or more input files, emitting the result
to standard out. If no files are provided, the digest of stdin will
be calculated.

`)
	flag.PrintDefaults()
}

func unsupported() {
	log.Fatalf("unsupported digest algorithm: %v", algorithm)
}

func main() {
	var jobs []job

	flag.Usage = usage
	flag.Parse()
	if showVersion {
		version.PrintVersion()
		return
	}

	var fail bool // if we fail on one item, foul the exit code
	if flag.NArg() > 0 {
		for _, path := range flag.Args() {
			fp, err := os.Open(path)

			if err != nil {
				log.Printf("%s: %v", path, err)
				fail = true
				continue
			}
			defer fp.Close()

			jobs = append(jobs, job{name: path, reader: fp})
		}
	} else {
		// just read stdin
		jobs = append(jobs, job{name: "-", reader: os.Stdin})
	}

	digestFn := algorithm.FromReader

	if !algorithm.Available() {
		unsupported()
	}

	for _, job := range jobs {
		dgst, err := digestFn(job.reader)
		if err != nil {
			log.Printf("%s: %v", job.name, err)
			fail = true
			continue
		}

		fmt.Printf("%v\t%s\n", dgst, job.name)
	}

	if fail {
		os.Exit(1)
	}
}
