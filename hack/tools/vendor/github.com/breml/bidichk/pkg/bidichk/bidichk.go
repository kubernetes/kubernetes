package bidichk

import (
	"bytes"
	"flag"
	"fmt"
	"go/token"
	"os"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
)

const (
	doc           = "bidichk detects dangerous unicode character sequences"
	disallowedDoc = `coma separated list of disallowed runes (full name or short name)

Supported runes

LEFT-TO-RIGHT-EMBEDDING, LRE (u+202A)
RIGHT-TO-LEFT-EMBEDDING, RLE (u+202B)
POP-DIRECTIONAL-FORMATTING, PDF (u+202C)
LEFT-TO-RIGHT-OVERRIDE, LRO (u+202D)
RIGHT-TO-LEFT-OVERRIDE, RLO (u+202E)
LEFT-TO-RIGHT-ISOLATE, LRI (u+2066)
RIGHT-TO-LEFT-ISOLATE, RLI (u+2067)
FIRST-STRONG-ISOLATE, FSI (u+2068)
POP-DIRECTIONAL-ISOLATE, PDI (u+2069)
`
)

type disallowedRunes map[string]rune

func (m disallowedRunes) String() string {
	ss := make([]string, 0, len(m))
	for s := range m {
		ss = append(ss, s)
	}
	sort.Strings(ss)
	return strings.Join(ss, ",")
}

func (m disallowedRunes) Set(s string) error {
	ss := strings.FieldsFunc(s, func(c rune) bool { return c == ',' })
	if len(ss) == 0 {
		return nil
	}

	for k := range m {
		delete(m, k)
	}

	for _, v := range ss {
		switch v {
		case runeShortNameLRE, runeShortNameRLE, runeShortNamePDF,
			runeShortNameLRO, runeShortNameRLO, runeShortNameLRI,
			runeShortNameRLI, runeShortNameFSI, runeShortNamePDI:
			v = shortNameLookup[v]
			fallthrough
		case runeNameLRE, runeNameRLE, runeNamePDF,
			runeNameLRO, runeNameRLO, runeNameLRI,
			runeNameRLI, runeNameFSI, runeNamePDI:
			m[v] = runeLookup[v]
		default:
			return fmt.Errorf("unknown check name %q (see help for full list)", v)
		}
	}
	return nil
}

const (
	runeNameLRE = "LEFT-TO-RIGHT-EMBEDDING"
	runeNameRLE = "RIGHT-TO-LEFT-EMBEDDING"
	runeNamePDF = "POP-DIRECTIONAL-FORMATTING"
	runeNameLRO = "LEFT-TO-RIGHT-OVERRIDE"
	runeNameRLO = "RIGHT-TO-LEFT-OVERRIDE"
	runeNameLRI = "LEFT-TO-RIGHT-ISOLATE"
	runeNameRLI = "RIGHT-TO-LEFT-ISOLATE"
	runeNameFSI = "FIRST-STRONG-ISOLATE"
	runeNamePDI = "POP-DIRECTIONAL-ISOLATE"

	runeShortNameLRE = "LRE" // LEFT-TO-RIGHT-EMBEDDING
	runeShortNameRLE = "RLE" // RIGHT-TO-LEFT-EMBEDDING
	runeShortNamePDF = "PDF" // POP-DIRECTIONAL-FORMATTING
	runeShortNameLRO = "LRO" // LEFT-TO-RIGHT-OVERRIDE
	runeShortNameRLO = "RLO" // RIGHT-TO-LEFT-OVERRIDE
	runeShortNameLRI = "LRI" // LEFT-TO-RIGHT-ISOLATE
	runeShortNameRLI = "RLI" // RIGHT-TO-LEFT-ISOLATE
	runeShortNameFSI = "FSI" // FIRST-STRONG-ISOLATE
	runeShortNamePDI = "PDI" // POP-DIRECTIONAL-ISOLATE
)

var runeLookup = map[string]rune{
	runeNameLRE: '\u202A', // LEFT-TO-RIGHT-EMBEDDING
	runeNameRLE: '\u202B', // RIGHT-TO-LEFT-EMBEDDING
	runeNamePDF: '\u202C', // POP-DIRECTIONAL-FORMATTING
	runeNameLRO: '\u202D', // LEFT-TO-RIGHT-OVERRIDE
	runeNameRLO: '\u202E', // RIGHT-TO-LEFT-OVERRIDE
	runeNameLRI: '\u2066', // LEFT-TO-RIGHT-ISOLATE
	runeNameRLI: '\u2067', // RIGHT-TO-LEFT-ISOLATE
	runeNameFSI: '\u2068', // FIRST-STRONG-ISOLATE
	runeNamePDI: '\u2069', // POP-DIRECTIONAL-ISOLATE
}

var shortNameLookup = map[string]string{
	runeShortNameLRE: runeNameLRE,
	runeShortNameRLE: runeNameRLE,
	runeShortNamePDF: runeNamePDF,
	runeShortNameLRO: runeNameLRO,
	runeShortNameRLO: runeNameRLO,
	runeShortNameLRI: runeNameLRI,
	runeShortNameRLI: runeNameRLI,
	runeShortNameFSI: runeNameFSI,
	runeShortNamePDI: runeNamePDI,
}

type bidichk struct {
	disallowedRunes disallowedRunes
}

// NewAnalyzer return a new bidichk analyzer.
func NewAnalyzer() *analysis.Analyzer {
	bidichk := bidichk{}
	bidichk.disallowedRunes = make(map[string]rune, len(runeLookup))
	for k, v := range runeLookup {
		bidichk.disallowedRunes[k] = v
	}

	a := &analysis.Analyzer{
		Name: "bidichk",
		Doc:  doc,
		Run:  bidichk.run,
	}

	a.Flags.Init("bidichk", flag.ExitOnError)
	a.Flags.Var(&bidichk.disallowedRunes, "disallowed-runes", disallowedDoc)
	a.Flags.Var(versionFlag{}, "V", "print version and exit")

	return a
}

func (b bidichk) run(pass *analysis.Pass) (interface{}, error) {
	var err error

	pass.Fset.Iterate(func(f *token.File) bool {
		if strings.HasPrefix(f.Name(), "$GOROOT") {
			return true
		}

		return b.check(f.Name(), f.Pos(0), pass) == nil
	})

	return nil, err
}

func (b bidichk) check(filename string, pos token.Pos, pass *analysis.Pass) error {
	body, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	for name, r := range b.disallowedRunes {
		start := 0
		for {
			idx := bytes.IndexRune(body[start:], r)
			if idx == -1 {
				break
			}
			start += idx

			pass.Reportf(pos+token.Pos(start), "found dangerous unicode character sequence %s", name)

			start += utf8.RuneLen(r)
		}
	}

	return nil
}
