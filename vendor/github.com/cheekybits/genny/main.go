package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/cheekybits/genny/out"
	"github.com/cheekybits/genny/parse"
)

/*

  source | genny gen [-in=""] [-out=""] [-pkg=""] "KeyType=string,int ValueType=string,int"

*/

const (
	_ = iota
	exitcodeInvalidArgs
	exitcodeInvalidTypeSet
	exitcodeStdinFailed
	exitcodeGenFailed
	exitcodeGetFailed
	exitcodeSourceFileInvalid
	exitcodeDestFileFailed
)

func main() {
	var (
		in      = flag.String("in", "", "file to parse instead of stdin")
		out     = flag.String("out", "", "file to save output to instead of stdout")
		pkgName = flag.String("pkg", "", "package name for generated files")
		prefix  = "https://github.com/metabition/gennylib/raw/master/"
	)
	flag.Parse()
	args := flag.Args()

	if len(args) < 2 {
		usage()
		os.Exit(exitcodeInvalidArgs)
	}

	if strings.ToLower(args[0]) != "gen" && strings.ToLower(args[0]) != "get" {
		usage()
		os.Exit(exitcodeInvalidArgs)
	}

	// parse the typesets
	var setsArg = args[1]
	if strings.ToLower(args[0]) == "get" {
		setsArg = args[2]
	}
	typeSets, err := parse.TypeSet(setsArg)
	if err != nil {
		fatal(exitcodeInvalidTypeSet, err)
	}

	outWriter := newWriter(*out)

	if strings.ToLower(args[0]) == "get" {
		if len(args) != 3 {
			fmt.Println("not enough arguments to get")
			usage()
			os.Exit(exitcodeInvalidArgs)
		}
		r, err := http.Get(prefix + args[1])
		if err != nil {
			fatal(exitcodeGetFailed, err)
		}
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			fatal(exitcodeGetFailed, err)
		}
		r.Body.Close()
		br := bytes.NewReader(b)
		err = gen(*in, *pkgName, br, typeSets, outWriter)
	} else if len(*in) > 0 {
		var file *os.File
		file, err = os.Open(*in)
		if err != nil {
			fatal(exitcodeSourceFileInvalid, err)
		}
		defer file.Close()
		err = gen(*in, *pkgName, file, typeSets, outWriter)
	} else {
		var source []byte
		source, err = ioutil.ReadAll(os.Stdin)
		if err != nil {
			fatal(exitcodeStdinFailed, err)
		}
		reader := bytes.NewReader(source)
		err = gen("stdin", *pkgName, reader, typeSets, outWriter)
	}

	// do the work
	if err != nil {
		fatal(exitcodeGenFailed, err)
	}

}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: genny [{flags}] gen "{types}"

gen - generates type specific code from generic code.
get <package/file> - fetch a generic template from the online library and gen it.

{flags}  - (optional) Command line flags (see below)
{types}  - (required) Specific types for each generic type in the source
{types} format:  {generic}={specific}[,another][ {generic2}={specific2}]

Examples:
  Generic=Specific
  Generic1=Specific1 Generic2=Specific2
  Generic1=Specific1,Specific2 Generic2=Specific3,Specific4

Flags:`)
	flag.PrintDefaults()
}

func newWriter(fileName string) io.Writer {
	if fileName == "" {
		return os.Stdout
	}
	lf := &out.LazyFile{FileName: fileName}
	defer lf.Close()
	return lf
}

func fatal(code int, a ...interface{}) {
	fmt.Println(a...)
	os.Exit(code)
}

// gen performs the generic generation.
func gen(filename, pkgName string, in io.ReadSeeker, typesets []map[string]string, out io.Writer) error {

	var output []byte
	var err error

	output, err = parse.Generics(filename, pkgName, in, typesets)
	if err != nil {
		return err
	}

	out.Write(output)
	return nil
}
