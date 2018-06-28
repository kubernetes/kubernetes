// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/dep/gps/pkgtree"
)

const (
	successChar   = "✓"
	successCharSp = successChar + " "
	failChar      = "✗"
	failCharSp    = failChar + " "
	backChar      = "←"
	innerIndent   = "  "
)

func (s *solver) traceCheckPkgs(bmi bimodalIdentifier) {
	if s.tl == nil {
		return
	}

	prefix := getprei(len(s.vqs) + 1)
	s.tl.Printf("%s\n", tracePrefix(fmt.Sprintf("? revisit %s to add %v pkgs", bmi.id, len(bmi.pl)), prefix, prefix))
}

func (s *solver) traceCheckQueue(q *versionQueue, bmi bimodalIdentifier, cont bool, offset int) {
	if s.tl == nil {
		return
	}

	prefix := getprei(len(s.vqs) + offset)
	vlen := strconv.Itoa(len(q.pi))
	if !q.allLoaded {
		vlen = "at least " + vlen
	}

	// TODO(sdboyer) how...to list the packages in the limited space we have?
	var verb string
	indent := ""
	if cont {
		// Continue is an "inner" message.. indenting
		verb = "continue"
		vlen = vlen + " more"
		indent = innerIndent
	} else {
		verb = "attempt"
	}

	s.tl.Printf("%s\n", tracePrefix(fmt.Sprintf("%s? %s %s with %v pkgs; %s versions to try", indent, verb, bmi.id, len(bmi.pl), vlen), prefix, prefix))
}

// traceStartBacktrack is called with the bmi that first failed, thus initiating
// backtracking
func (s *solver) traceStartBacktrack(bmi bimodalIdentifier, err error, pkgonly bool) {
	if s.tl == nil {
		return
	}

	var msg string
	if pkgonly {
		msg = fmt.Sprintf("%s%s could not add %v pkgs to %s; begin backtrack", innerIndent, backChar, len(bmi.pl), bmi.id)
	} else {
		msg = fmt.Sprintf("%s%s no more versions of %s to try; begin backtrack", innerIndent, backChar, bmi.id)
	}

	prefix := getprei(len(s.sel.projects))
	s.tl.Printf("%s\n", tracePrefix(msg, prefix, prefix))
}

// traceBacktrack is called when a package or project is poppped off during
// backtracking
func (s *solver) traceBacktrack(bmi bimodalIdentifier, pkgonly bool) {
	if s.tl == nil {
		return
	}

	var msg string
	if pkgonly {
		msg = fmt.Sprintf("%s backtrack: popped %v pkgs from %s", backChar, len(bmi.pl), bmi.id)
	} else {
		msg = fmt.Sprintf("%s backtrack: no more versions of %s to try", backChar, bmi.id)
	}

	prefix := getprei(len(s.sel.projects))
	s.tl.Printf("%s\n", tracePrefix(msg, prefix, prefix))
}

// Called just once after solving has finished, whether success or not
func (s *solver) traceFinish(sol solution, err error) {
	if s.tl == nil {
		return
	}

	if err == nil {
		var pkgcount int
		for _, lp := range sol.Projects() {
			pkgcount += len(lp.pkgs)
		}
		s.tl.Printf("%s%s found solution with %v packages from %v projects", innerIndent, successChar, pkgcount, len(sol.Projects()))
	} else {
		s.tl.Printf("%s%s solving failed", innerIndent, failChar)
	}
}

// traceSelectRoot is called just once, when the root project is selected
func (s *solver) traceSelectRoot(ptree pkgtree.PackageTree, cdeps []completeDep) {
	if s.tl == nil {
		return
	}

	// This duplicates work a bit, but we're in trace mode and it's only once,
	// so who cares
	rm, _ := ptree.ToReachMap(true, true, false, s.rd.ir)

	s.tl.Printf("Root project is %q", s.rd.rpt.ImportRoot)

	var expkgs int
	for _, cdep := range cdeps {
		expkgs += len(cdep.pl)
	}

	// TODO(sdboyer) include info on ignored pkgs/imports, etc.
	s.tl.Printf(" %v transitively valid internal packages", len(rm))
	s.tl.Printf(" %v external packages imported from %v projects", expkgs, len(cdeps))
	s.tl.Printf("(0)   " + successCharSp + "select (root)")
}

// traceSelect is called when an atom is successfully selected
func (s *solver) traceSelect(awp atomWithPackages, pkgonly bool) {
	if s.tl == nil {
		return
	}

	var msg string
	if pkgonly {
		msg = fmt.Sprintf("%s%s include %v more pkgs from %s", innerIndent, successChar, len(awp.pl), a2vs(awp.a))
	} else {
		msg = fmt.Sprintf("%s select %s w/%v pkgs", successChar, a2vs(awp.a), len(awp.pl))
	}

	prefix := getprei(len(s.sel.projects) - 1)
	s.tl.Printf("%s\n", tracePrefix(msg, prefix, prefix))
}

func (s *solver) traceInfo(args ...interface{}) {
	if s.tl == nil {
		return
	}

	if len(args) == 0 {
		panic("must pass at least one param to traceInfo")
	}

	preflen := len(s.sel.projects)
	var msg string
	switch data := args[0].(type) {
	case string:
		msg = tracePrefix(innerIndent+fmt.Sprintf(data, args[1:]...), "  ", "  ")
	case traceError:
		preflen++
		// We got a special traceError, use its custom method
		msg = tracePrefix(innerIndent+data.traceString(), "  ", failCharSp)
	case error:
		// Regular error; still use the x leader but default Error() string
		msg = tracePrefix(innerIndent+data.Error(), "  ", failCharSp)
	default:
		// panic here because this can *only* mean a stupid internal bug
		panic(fmt.Sprintf("canary - unknown type passed as first param to traceInfo %T", data))
	}

	prefix := getprei(preflen)
	s.tl.Printf("%s\n", tracePrefix(msg, prefix, prefix))
}

func getprei(i int) string {
	var s string
	if i < 10 {
		s = fmt.Sprintf("(%d)	", i)
	} else if i < 100 {
		s = fmt.Sprintf("(%d)  ", i)
	} else {
		s = fmt.Sprintf("(%d) ", i)
	}
	return s
}

func tracePrefix(msg, sep, fsep string) string {
	parts := strings.Split(strings.TrimSuffix(msg, "\n"), "\n")
	for k, str := range parts {
		if k == 0 {
			parts[k] = fsep + str
		} else {
			parts[k] = sep + str
		}
	}

	return strings.Join(parts, "\n")
}
