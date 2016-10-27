// This work is subject to the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
// license. Its contents can be found at:
// http://creativecommons.org/publicdomain/zero/1.0/

package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/jteeuwen/go-bindata"
)

func main() {
	cfg := parseArgs()
	err := bindata.Translate(cfg)

	if err != nil {
		fmt.Fprintf(os.Stderr, "bindata: %v\n", err)
		os.Exit(1)
	}
}

// parseArgs create s a new, filled configuration instance
// by reading and parsing command line options.
//
// This function exits the program with an error, if
// any of the command line options are incorrect.
func parseArgs() *bindata.Config {
	var version bool

	c := bindata.NewConfig()

	flag.Usage = func() {
		fmt.Printf("Usage: %s [options] <input directories>\n\n", os.Args[0])
		flag.PrintDefaults()
	}

	flag.BoolVar(&c.Debug, "debug", c.Debug, "Do not embed the assets, but provide the embedding API. Contents will still be loaded from disk.")
	flag.BoolVar(&c.Dev, "dev", c.Dev, "Similar to debug, but does not emit absolute paths. Expects a rootDir variable to already exist in the generated code's package.")
	flag.StringVar(&c.Tags, "tags", c.Tags, "Optional set of build tags to include.")
	flag.StringVar(&c.Prefix, "prefix", c.Prefix, "Optional path prefix to strip off asset names.")
	flag.StringVar(&c.Package, "pkg", c.Package, "Package name to use in the generated code.")
	flag.BoolVar(&c.NoMemCopy, "nomemcopy", c.NoMemCopy, "Use a .rodata hack to get rid of unnecessary memcopies. Refer to the documentation to see what implications this carries.")
	flag.BoolVar(&c.NoCompress, "nocompress", c.NoCompress, "Assets will *not* be GZIP compressed when this flag is specified.")
	flag.BoolVar(&c.NoMetadata, "nometadata", c.NoMetadata, "Assets will not preserve size, mode, and modtime info.")
	flag.UintVar(&c.Mode, "mode", c.Mode, "Optional file mode override for all files.")
	flag.Int64Var(&c.ModTime, "modtime", c.ModTime, "Optional modification unix timestamp override for all files.")
	flag.StringVar(&c.Output, "o", c.Output, "Optional name of the output file to be generated.")
	flag.BoolVar(&version, "version", false, "Displays version information.")

	ignore := make([]string, 0)
	flag.Var((*AppendSliceValue)(&ignore), "ignore", "Regex pattern to ignore")

	flag.Parse()

	patterns := make([]*regexp.Regexp, 0)
	for _, pattern := range ignore {
		patterns = append(patterns, regexp.MustCompile(pattern))
	}
	c.Ignore = patterns

	if version {
		fmt.Printf("%s\n", Version())
		os.Exit(0)
	}

	// Make sure we have input paths.
	if flag.NArg() == 0 {
		fmt.Fprintf(os.Stderr, "Missing <input dir>\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Create input configurations.
	c.Input = make([]bindata.InputConfig, flag.NArg())
	for i := range c.Input {
		c.Input[i] = parseInput(flag.Arg(i))
	}

	return c
}

// parseRecursive determines whether the given path has a recrusive indicator and
// returns a new path with the recursive indicator chopped off if it does.
//
//  ex:
//      /path/to/foo/...    -> (/path/to/foo, true)
//      /path/to/bar        -> (/path/to/bar, false)
func parseInput(path string) bindata.InputConfig {
	if strings.HasSuffix(path, "/...") {
		return bindata.InputConfig{
			Path:      filepath.Clean(path[:len(path)-4]),
			Recursive: true,
		}
	} else {
		return bindata.InputConfig{
			Path:      filepath.Clean(path),
			Recursive: false,
		}
	}

}
