package dupl

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"

	"github.com/golangci/dupl/job"
	"github.com/golangci/dupl/printer"
	"github.com/golangci/dupl/syntax"
)

const defaultThreshold = 15

var (
	paths   = []string{"."}
	vendor  = flag.Bool("dupl.vendor", false, "")
	verbose = flag.Bool("dupl.verbose", false, "")
	files   = flag.Bool("dupl.files", false, "")

	html     = flag.Bool("dupl.html", false, "")
	plumbing = flag.Bool("dupl.plumbing", false, "")
)

const (
	vendorDirPrefix = "vendor" + string(filepath.Separator)
	vendorDirInPath = string(filepath.Separator) + vendorDirPrefix
)

func init() {
	flag.BoolVar(verbose, "dupl.v", false, "alias for -verbose")
}

func Run(files []string, threshold int) ([]printer.Issue, error) {
	fchan := make(chan string, 1024)
	go func() {
		for _, f := range files {
			fchan <- f
		}
		close(fchan)
	}()
	schan := job.Parse(fchan)
	t, data, done := job.BuildTree(schan)
	<-done

	// finish stream
	t.Update(&syntax.Node{Type: -1})

	mchan := t.FindDuplOver(threshold)
	duplChan := make(chan syntax.Match)
	go func() {
		for m := range mchan {
			match := syntax.FindSyntaxUnits(*data, m, threshold)
			if len(match.Frags) > 0 {
				duplChan <- match
			}
		}
		close(duplChan)
	}()

	return makeIssues(duplChan)
}

func makeIssues(duplChan <-chan syntax.Match) ([]printer.Issue, error) {
	groups := make(map[string][][]*syntax.Node)
	for dupl := range duplChan {
		groups[dupl.Hash] = append(groups[dupl.Hash], dupl.Frags...)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	p := printer.NewPlumbing(ioutil.ReadFile)

	var issues []printer.Issue
	for _, k := range keys {
		uniq := unique(groups[k])
		if len(uniq) > 1 {
			i, err := p.MakeIssues(uniq)
			if err != nil {
				return nil, err
			}
			issues = append(issues, i...)
		}
	}

	return issues, nil
}

func unique(group [][]*syntax.Node) [][]*syntax.Node {
	fileMap := make(map[string]map[int]struct{})

	var newGroup [][]*syntax.Node
	for _, seq := range group {
		node := seq[0]
		file, ok := fileMap[node.Filename]
		if !ok {
			file = make(map[int]struct{})
			fileMap[node.Filename] = file
		}
		if _, ok := file[node.Pos]; !ok {
			file[node.Pos] = struct{}{}
			newGroup = append(newGroup, seq)
		}
	}
	return newGroup
}

func usage() {
	fmt.Fprintln(os.Stderr, `Usage: dupl [flags] [paths]

Paths:
  If the given path is a file, dupl will use it regardless of
  the file extension. If it is a directory, it will recursively
  search for *.go files in that directory.

  If no path is given, dupl will recursively search for *.go
  files in the current directory.

Flags:
  -files
    	read file names from stdin one at each line
  -html
    	output the results as HTML, including duplicate code fragments
  -plumbing
    	plumbing (easy-to-parse) output for consumption by scripts or tools
  -t, -threshold size
    	minimum token sequence size as a clone (default 15)
  -vendor
    	check files in vendor directory
  -v, -verbose
    	explain what is being done

Examples:
  dupl -t 100
    	Search clones in the current directory of size at least
    	100 tokens.
  dupl $(find app/ -name '*_test.go')
    	Search for clones in tests in the app directory.
  find app/ -name '*_test.go' |dupl -files
    	The same as above.`)
	os.Exit(2)
}
