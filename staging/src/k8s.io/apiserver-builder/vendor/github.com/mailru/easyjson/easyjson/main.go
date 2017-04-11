package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/mailru/easyjson/bootstrap"
	// Reference the gen package to be friendly to vendoring tools,
	// as it is an indirect dependency.
	// (The temporary bootstrapping code uses it.)
	_ "github.com/mailru/easyjson/gen"
	"github.com/mailru/easyjson/parser"
)

var buildTags = flag.String("build_tags", "", "build tags to add to generated file")
var snakeCase = flag.Bool("snake_case", false, "use snake_case names instead of CamelCase by default")
var noStdMarshalers = flag.Bool("no_std_marshalers", false, "don't generate MarshalJSON/UnmarshalJSON methods")
var omitEmpty = flag.Bool("omit_empty", false, "omit empty fields by default")
var allStructs = flag.Bool("all", false, "generate un-/marshallers for all structs in a file")
var leaveTemps = flag.Bool("leave_temps", false, "do not delete temporary files")
var stubs = flag.Bool("stubs", false, "only generate stubs for marshallers/unmarshallers methods")
var noformat = flag.Bool("noformat", false, "do not run 'gofmt -w' on output file")
var specifiedName = flag.String("output_filename", "", "specify the filename of the output")

func generate(fname string) (err error) {
	p := parser.Parser{AllStructs: *allStructs}
	if err := p.Parse(fname); err != nil {
		return fmt.Errorf("Error parsing %v: %v", fname, err)
	}

	var outName string
	if s := strings.TrimSuffix(fname, ".go"); s == fname {
		return fmt.Errorf("Filename must end in '.go'")
	} else {
		outName = s + "_easyjson.go"
	}

	if *specifiedName != "" {
		outName = *specifiedName
	}

	g := bootstrap.Generator{
		BuildTags:       *buildTags,
		PkgPath:         p.PkgPath,
		PkgName:         p.PkgName,
		Types:           p.StructNames,
		SnakeCase:       *snakeCase,
		NoStdMarshalers: *noStdMarshalers,
		OmitEmpty:       *omitEmpty,
		LeaveTemps:      *leaveTemps,
		OutName:         outName,
		StubsOnly:       *stubs,
		NoFormat:        *noformat,
	}

	if err := g.Run(); err != nil {
		return fmt.Errorf("Bootstrap failed: %v", err)
	}
	return nil
}

func main() {
	flag.Parse()

	files := flag.Args()

	gofile := os.Getenv("GOFILE")
	if len(files) == 0 && gofile != "" {
		files = []string{gofile}
	} else if len(files) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	for _, fname := range files {
		if err := generate(fname); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
