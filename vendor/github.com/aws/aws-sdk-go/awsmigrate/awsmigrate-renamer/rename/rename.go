// +build go1.5,deprecated

package rename

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"

	"golang.org/x/tools/go/loader"
)

var dryRun = flag.Bool("dryrun", false, "Dry run")
var verbose = flag.Bool("verbose", false, "Verbose")

type packageRenames struct {
	operations map[string]string
	shapes     map[string]string
	fields     map[string]string
}

type renamer struct {
	*loader.Program
	files map[*token.File]bool
}

// ParsePathsFromArgs parses arguments from command line and looks at import
// paths to rename objects.
func ParsePathsFromArgs() {
	flag.Parse()
	for _, dir := range flag.Args() {
		var conf loader.Config
		conf.ParserMode = parser.ParseComments
		conf.ImportWithTests(dir)
		prog, err := conf.Load()
		if err != nil {
			panic(err)
		}

		r := renamer{prog, map[*token.File]bool{}}
		r.parse()
		if !*dryRun {
			r.write()
		}
	}
}

func (r *renamer) dryInfo() string {
	if *dryRun {
		return "[DRY-RUN]"
	}
	return "[!]"
}

func (r *renamer) printf(msg string, args ...interface{}) {
	if *verbose {
		fmt.Printf(msg, args...)
	}
}

func (r *renamer) parse() {
	for _, pkg := range r.InitialPackages() {
		r.parseUses(pkg)
	}
}

func (r *renamer) write() {
	for _, pkg := range r.InitialPackages() {
		for _, f := range pkg.Files {
			tokenFile := r.Fset.File(f.Pos())
			if r.files[tokenFile] {
				var buf bytes.Buffer
				format.Node(&buf, r.Fset, f)
				if err := ioutil.WriteFile(tokenFile.Name(), buf.Bytes(), 0644); err != nil {
					panic(err)
				}
			}
		}
	}
}

func (r *renamer) parseUses(pkg *loader.PackageInfo) {
	for k, v := range pkg.Uses {
		if v.Pkg() != nil {
			pkgPath := v.Pkg().Path()
			if renames, ok := renamedPackages[pkgPath]; ok {
				name := k.Name
				switch t := v.(type) {
				case *types.Func:
					if newName, ok := renames.operations[t.Name()]; ok && newName != name {
						r.printf("%s Rename [OPERATION]: %q -> %q\n", r.dryInfo(), name, newName)
						r.files[r.Fset.File(k.Pos())] = true
						k.Name = newName
					}
				case *types.TypeName:
					if newName, ok := renames.shapes[name]; ok && newName != name {
						r.printf("%s Rename [SHAPE]: %q -> %q\n", r.dryInfo(), t.Name(), newName)
						r.files[r.Fset.File(k.Pos())] = true
						k.Name = newName
					}
				case *types.Var:
					if newName, ok := renames.fields[name]; ok && newName != name {
						r.printf("%s Rename [FIELD]: %q -> %q\n", r.dryInfo(), t.Name(), newName)
						r.files[r.Fset.File(k.Pos())] = true
						k.Name = newName
					}
				}
			}
		}
	}
}
