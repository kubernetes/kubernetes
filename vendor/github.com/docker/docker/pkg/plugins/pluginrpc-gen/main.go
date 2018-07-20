package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"os"
	"unicode"
	"unicode/utf8"
)

type stringSet struct {
	values map[string]struct{}
}

func (s stringSet) String() string {
	return ""
}

func (s stringSet) Set(value string) error {
	s.values[value] = struct{}{}
	return nil
}
func (s stringSet) GetValues() map[string]struct{} {
	return s.values
}

var (
	typeName   = flag.String("type", "", "interface type to generate plugin rpc proxy for")
	rpcName    = flag.String("name", *typeName, "RPC name, set if different from type")
	inputFile  = flag.String("i", "", "input file path")
	outputFile = flag.String("o", *inputFile+"_proxy.go", "output file path")

	skipFuncs   map[string]struct{}
	flSkipFuncs = stringSet{make(map[string]struct{})}

	flBuildTags = stringSet{make(map[string]struct{})}
)

func errorOut(msg string, err error) {
	if err == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "%s: %v\n", msg, err)
	os.Exit(1)
}

func checkFlags() error {
	if *outputFile == "" {
		return fmt.Errorf("missing required flag `-o`")
	}
	if *inputFile == "" {
		return fmt.Errorf("missing required flag `-i`")
	}
	return nil
}

func main() {
	flag.Var(flSkipFuncs, "skip", "skip parsing for function")
	flag.Var(flBuildTags, "tag", "build tags to add to generated files")
	flag.Parse()
	skipFuncs = flSkipFuncs.GetValues()

	errorOut("error", checkFlags())

	pkg, err := Parse(*inputFile, *typeName)
	errorOut(fmt.Sprintf("error parsing requested type %s", *typeName), err)

	var analysis = struct {
		InterfaceType string
		RPCName       string
		BuildTags     map[string]struct{}
		*ParsedPkg
	}{toLower(*typeName), *rpcName, flBuildTags.GetValues(), pkg}
	var buf bytes.Buffer

	errorOut("parser error", generatedTempl.Execute(&buf, analysis))
	src, err := format.Source(buf.Bytes())
	errorOut("error formatting generated source:\n"+buf.String(), err)
	errorOut("error writing file", ioutil.WriteFile(*outputFile, src, 0644))
}

func toLower(s string) string {
	if s == "" {
		return ""
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToLower(r)) + s[n:]
}
