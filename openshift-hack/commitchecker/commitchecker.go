package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	var start, end string
	flag.StringVar(&start, "start", "master", "The start of the revision range for analysis")
	flag.StringVar(&end, "end", "HEAD", "The end of the revision range for analysis")
	flag.Parse()

	commits, err := CommitsBetween(start, end)
	if err != nil {
		if err == ErrNotCommit {
			_, _ = fmt.Fprintf(os.Stderr, "WARNING: one of the provided commits does not exist, not a true branch\n")
			os.Exit(0)
		}
		_, _ = fmt.Fprintf(os.Stderr, "ERROR: couldn't find commits from %s..%s: %v\n", start, end, err)
		os.Exit(1)
	}

	var errs []string
	for _, validate := range AllCommitValidators {
		for _, commit := range commits {
			errs = append(errs, validate(commit)...)
		}
	}

	if len(errs) > 0 {
		for _, e := range errs {
			_, _ = fmt.Fprintf(os.Stderr, "%s\n\n", e)
		}

		os.Exit(2)
	}
}
