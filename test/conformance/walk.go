/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package main provides a tool that scans kubernetes e2e test source code
// looking for conformance test declarations, which it emits on stdout.  It
// also looks for legacy, manually added "[Conformance]" tags and reports an
// error if it finds any.
//
// This approach is not air tight, but it will serve our purpose as a
// pre-submit check.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"text/template"
)

var (
	baseURL                                           = flag.String("url", "https://github.com/kubernetes/kubernetes/tree/master/", "location of the current source")
	confDoc                                           = flag.Bool("conformance", false, "write a conformance document")
	version                                           = flag.String("version", "v1.9", "version of this conformance document")
	totalConfTests, totalLegacyTests, missingComments int

	// If a test name contains any of these tags, it is ineligble for promotion to conformance
	regexIneligibleTags = regexp.MustCompile(`\[(Alpha|Disruptive|Feature:[^\]]+|Flaky)\]`)
)

const regexDescribe = "Describe|KubeDescribe|SIGDescribe"
const regexContext = "^Context$"

type visitor struct {
	FileSet   *token.FileSet
	describes []describe
	cMap      ast.CommentMap
	//list of all the conformance tests in the path
	tests []conformanceData
}

//describe contains text associated with ginkgo describe container
type describe struct {
	rparen      token.Pos
	text        string
	lastContext context
}

//context contain the text associated with the Context clause
type context struct {
	text string
}

type conformanceData struct {
	// A URL to the line of code in the kube src repo for the test
	URL string
	// Extracted from the "Testname:" comment before the test
	TestName string
	// Extracted from the "Description:" comment before the test
	Description string
	// Version when this test is added or modified ex: v1.12, v1.13
	Release string
}

func (v *visitor) convertToConformanceData(at *ast.BasicLit) {
	cd := conformanceData{}

	comment := v.comment(at)
	pos := v.FileSet.Position(at.Pos())
	cd.URL = fmt.Sprintf("%s%s#L%d", *baseURL, pos.Filename, pos.Line)

	lines := strings.Split(comment, "\n")
	cd.Description = ""
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if sline := regexp.MustCompile("^Testname\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			cd.TestName = sline[1]
			continue
		}
		if sline := regexp.MustCompile("^Release\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			cd.Release = sline[1]
			continue
		}
		if sline := regexp.MustCompile("^Description\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			line = sline[1]
		}
		cd.Description += line + "\n"
	}

	if cd.TestName == "" {
		testName := v.getDescription(at.Value)
		i := strings.Index(testName, "[Conformance]")
		if i > 0 {
			cd.TestName = strings.TrimSpace(testName[:i])
		} else {
			cd.TestName = testName
		}
	}

	v.tests = append(v.tests, cd)
}

func newVisitor() *visitor {
	return &visitor{
		FileSet: token.NewFileSet(),
	}
}

func (v *visitor) isConformanceCall(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		if fun.Sel != nil {
			return fun.Sel.Name == "ConformanceIt"
		}
	}
	return false
}

func (v *visitor) isLegacyItCall(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		if fun.Name != "It" {
			return false
		}
		if len(call.Args) < 1 {
			v.failf(call, "Not enough arguments to It()")
		}
	default:
		return false
	}

	switch arg := call.Args[0].(type) {
	case *ast.BasicLit:
		if arg.Kind != token.STRING {
			v.failf(arg, "Unexpected non-string argument to It()")
		}
		if strings.Contains(arg.Value, "[Conformance]") {
			return true
		}
	default:
		// non-literal argument to It()... we just ignore these even though they could be a way to "sneak in" a conformance test
	}

	return false
}

func (v *visitor) failf(expr ast.Expr, format string, a ...interface{}) {
	msg := fmt.Sprintf(format, a...)
	fmt.Fprintf(os.Stderr, "ERROR at %v: %s\n", v.FileSet.Position(expr.Pos()), msg)
}

func (v *visitor) comment(x *ast.BasicLit) string {
	for _, comm := range v.cMap.Comments() {
		testOffset := int(x.Pos()-comm.End()) - len("framework.ConformanceIt(\"")
		//Cannot assume the offset is within three or four tabs from the test block itself.
		//It is better to trim the newlines, tabs, etc and then we if the comment is followed
		//by the test block itself so that we can associate the comment with it properly.
		if 0 <= testOffset && testOffset <= 10 {
			b1 := make([]byte, x.Pos()-comm.End())
			//if we fail to open the file to compare the content we just assume the
			//proximity of the comment and apply it.
			myf, err := os.Open(v.FileSet.File(x.Pos()).Name())
			if err == nil {
				if _, err := myf.Seek(int64(comm.End()), 0); err == nil {
					if _, err := myf.Read(b1); err == nil {
						if strings.Compare(strings.Trim(string(b1), "\t \r\n"), "framework.ConformanceIt(\"") == 0 {
							return comm.Text()
						}
					}
				}
			} else {
				//comment section's end is noticed within 10 characters from framework.ConformanceIt block
				return comm.Text()
			}
		}
	}
	return ""
}

func (v *visitor) emit(arg ast.Expr) {
	switch at := arg.(type) {
	case *ast.BasicLit:
		if at.Kind != token.STRING {
			v.failf(at, "framework.ConformanceIt() called with non-string argument")
			return
		}

		description := v.getDescription(at.Value)
		err := validateTestName(description)
		if err != nil {
			v.failf(at, err.Error())
			return
		}

		at.Value = normalizeTestName(at.Value)
		if *confDoc {
			v.convertToConformanceData(at)
		} else {
			fmt.Printf("%s: %q\n", v.FileSet.Position(at.Pos()).Filename, at.Value)
		}
	default:
		v.failf(at, "framework.ConformanceIt() called with non-literal argument")
		fmt.Fprintf(os.Stderr, "ERROR: non-literal argument %v at %v\n", arg, v.FileSet.Position(arg.Pos()))
	}
}

func (v *visitor) getDescription(value string) string {
	tokens := []string{}
	for _, describe := range v.describes {
		tokens = append(tokens, describe.text)
		if len(describe.lastContext.text) > 0 {
			tokens = append(tokens, describe.lastContext.text)
		}
	}
	tokens = append(tokens, value)

	trimmed := []string{}
	for _, token := range tokens {
		trimmed = append(trimmed, strings.Trim(token, "\""))
	}

	return strings.Join(trimmed, " ")
}

var (
	regexTag = regexp.MustCompile(`(\[[a-zA-Z0-9:-]+\])`)
)

// normalizeTestName removes tags (e.g., [Feature:Foo]), double quotes and trim
// the spaces to normalize the test name.
func normalizeTestName(s string) string {
	r := regexTag.ReplaceAllString(s, "")
	r = strings.Trim(r, "\"")
	return strings.TrimSpace(r)
}

func validateTestName(s string) error {
	matches := regexIneligibleTags.FindAllString(s, -1)
	if matches != nil {
		return fmt.Errorf("'%s' cannot have invalid tags %v", s, strings.Join(matches, ","))
	}
	return nil
}

// funcName converts a selectorExpr with two idents into a string,
// x.y -> "x.y"
func funcName(n ast.Expr) string {
	if sel, ok := n.(*ast.SelectorExpr); ok {
		if x, ok := sel.X.(*ast.Ident); ok {
			return x.String() + "." + sel.Sel.String()
		}
	}
	return ""
}

// isSprintf returns whether the given node is a call to fmt.Sprintf
func isSprintf(n ast.Expr) bool {
	call, ok := n.(*ast.CallExpr)
	return ok && funcName(call.Fun) == "fmt.Sprintf" && len(call.Args) != 0
}

// firstArg attempts to statically determine the value of the first
// argument. It only handles strings, and converts any unknown values
// (fmt.Sprintf interpolations) into *.
func (v *visitor) firstArg(n *ast.CallExpr) string {
	if len(n.Args) == 0 {
		return ""
	}
	var lit *ast.BasicLit
	if isSprintf(n.Args[0]) {
		return v.firstArg(n.Args[0].(*ast.CallExpr))
	}
	lit, ok := n.Args[0].(*ast.BasicLit)
	if ok && lit.Kind == token.STRING {
		val, err := strconv.Unquote(lit.Value)
		if err != nil {
			panic(err)
		}
		if strings.Contains(val, "%") {
			val = strings.Replace(val, "%d", "*", -1)
			val = strings.Replace(val, "%v", "*", -1)
			val = strings.Replace(val, "%s", "*", -1)
		}
		return val
	}
	if ident, ok := n.Args[0].(*ast.Ident); ok {
		return ident.String()
	}
	return "*"
}

// matchFuncName returns the first argument of a function if it's
// a Ginkgo-relevant function (Describe/KubeDescribe/Context),
// and the empty string otherwise.
func (v *visitor) matchFuncName(n *ast.CallExpr, pattern string) string {
	switch x := n.Fun.(type) {
	case *ast.SelectorExpr:
		if match, err := regexp.MatchString(pattern, x.Sel.Name); err == nil && match {
			return v.firstArg(n)
		}
	case *ast.Ident:
		if match, err := regexp.MatchString(pattern, x.Name); err == nil && match {
			return v.firstArg(n)
		}
	default:
		return ""
	}
	return ""
}

// Visit visits each node looking for either calls to framework.ConformanceIt,
// which it will emit in its list of conformance tests, or legacy calls to
// It() with a manually embedded [Conformance] tag, which it will complain
// about.
func (v *visitor) Visit(node ast.Node) (w ast.Visitor) {
	lastDescribe := len(v.describes) - 1

	switch t := node.(type) {
	case *ast.CallExpr:
		if name := v.matchFuncName(t, regexDescribe); name != "" && len(t.Args) >= 2 {
			v.describes = append(v.describes, describe{text: name, rparen: t.Rparen})
		} else if name := v.matchFuncName(t, regexContext); name != "" && len(t.Args) >= 2 {
			if lastDescribe > -1 {
				v.describes[lastDescribe].lastContext = context{text: name}
			}
		} else if v.isConformanceCall(t) {
			totalConfTests++
			v.emit(t.Args[0])
			return nil
		} else if v.isLegacyItCall(t) {
			totalLegacyTests++
			v.failf(t, "Using It() with manual [Conformance] tag is no longer allowed.  Use framework.ConformanceIt() instead.")
			return nil
		}
	}

	// If we're past the position of the last describe's rparen, pop the describe off
	if lastDescribe > -1 && node != nil {
		if node.Pos() > v.describes[lastDescribe].rparen {
			v.describes = v.describes[:lastDescribe]
		}
	}

	return v
}

func scanfile(path string, src interface{}) []conformanceData {
	v := newVisitor()
	file, err := parser.ParseFile(v.FileSet, path, src, parser.ParseComments)
	if err != nil {
		panic(err)
	}

	v.cMap = ast.NewCommentMap(v.FileSet, file, file.Comments)

	ast.Walk(v, file)
	return v.tests
}

func main() {
	flag.Parse()

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "USAGE: %s <DIR or FILE> [...]\n", os.Args[0])
		os.Exit(64)
	}

	if *confDoc {
		// Note: this assumes that you're running from the root of the kube src repo
		templ, err := template.ParseFiles("test/conformance/cf_header.md")
		if err != nil {
			fmt.Printf("Error reading the Header file information: %s\n\n", err)
		}
		data := struct {
			Version string
		}{
			Version: *version,
		}
		templ.Execute(os.Stdout, data)
	}

	totalConfTests = 0
	totalLegacyTests = 0
	missingComments = 0
	for _, arg := range flag.Args() {
		filepath.Walk(arg, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if strings.HasSuffix(path, ".go") {
				tests := scanfile(path, nil)
				for _, cd := range tests {
					fmt.Printf("## [%s](%s)\n\n", cd.TestName, cd.URL)
					fmt.Printf("### Release %s\n", cd.Release)
					fmt.Printf("%s\n\n", cd.Description)
					if len(cd.Description) < 10 {
						missingComments++
					}
				}
			}
			return nil
		})
	}
	if *confDoc {
		fmt.Println("\n## **Summary**")
		fmt.Printf("\nTotal Conformance Tests: %d, total legacy tests that need conversion: %d, while total tests that need comment sections: %d\n\n", totalConfTests, totalLegacyTests, missingComments)
	}
}
