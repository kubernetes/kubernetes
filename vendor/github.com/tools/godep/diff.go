package main

import (
	"bytes"
	"fmt"
	"log"

	"github.com/pmezard/go-difflib/difflib"
)

var cmdDiff = &Command{
	Name:  "diff",
	Short: "shows the diff between current and previously saved set of dependencies",
	Long: `
Shows the difference, in a unified diff format, between the
current set of dependencies and those generated on a
previous 'go save' execution.
`,
	Run:          runDiff,
	OnlyInGOPATH: true,
}

func runDiff(cmd *Command, args []string) {
	gold, err := loadDefaultGodepsFile()
	if err != nil {
		log.Fatalln(err)
	}

	pkgs := []string{"."}
	dot, err := LoadPackages(pkgs...)
	if err != nil {
		log.Fatalln(err)
	}

	gnew := &Godeps{
		ImportPath: dot[0].ImportPath,
		GoVersion:  gold.GoVersion,
	}

	err = gnew.fill(dot, dot[0].ImportPath)
	if err != nil {
		log.Fatalln(err)
	}

	diff, err := diffStr(&gold, gnew)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(diff)
}

// diffStr returns a unified diff string of two Godeps.
func diffStr(a, b *Godeps) (string, error) {
	var ab, bb bytes.Buffer

	_, err := a.writeTo(&ab)
	if err != nil {
		log.Fatalln(err)
	}

	_, err = b.writeTo(&bb)
	if err != nil {
		log.Fatalln(err)
	}

	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(ab.String()),
		B:        difflib.SplitLines(bb.String()),
		FromFile: b.file(),
		ToFile:   "$GOPATH",
		Context:  10,
	}
	return difflib.GetUnifiedDiffString(diff)
}
