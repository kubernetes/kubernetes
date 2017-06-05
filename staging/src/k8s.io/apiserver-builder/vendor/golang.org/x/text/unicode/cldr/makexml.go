// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This tool generates types for the various XML formats of CLDR.
package main

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"

	"golang.org/x/text/internal/gen"
)

var outputFile = flag.String("output", "xml.go", "output file name")

func main() {
	flag.Parse()

	r := gen.OpenCLDRCoreZip()
	buffer, err := ioutil.ReadAll(r)
	if err != nil {
		log.Fatal("Could not read zip file")
	}
	r.Close()
	z, err := zip.NewReader(bytes.NewReader(buffer), int64(len(buffer)))
	if err != nil {
		log.Fatalf("Could not read zip archive: %v", err)
	}

	var buf bytes.Buffer

	version := gen.CLDRVersion()

	for _, dtd := range files {
		for _, f := range z.File {
			if strings.HasSuffix(f.Name, dtd.file+".dtd") {
				r, err := f.Open()
				failOnError(err)

				b := makeBuilder(&buf, dtd)
				b.parseDTD(r)
				b.resolve(b.index[dtd.top[0]])
				b.write()
				if b.version != "" && version != b.version {
					println(f.Name)
					log.Fatalf("main: inconsistent versions: found %s; want %s", b.version, version)
				}
				break
			}
		}
	}
	fmt.Fprintln(&buf, "// Version is the version of CLDR from which the XML definitions are generated.")
	fmt.Fprintf(&buf, "const Version = %q\n", version)

	gen.WriteGoFile(*outputFile, "cldr", buf.Bytes())
}

func failOnError(err error) {
	if err != nil {
		log.New(os.Stderr, "", log.Lshortfile).Output(2, err.Error())
		os.Exit(1)
	}
}

// configuration data per DTD type
type dtd struct {
	file string   // base file name
	root string   // Go name of the root XML element
	top  []string // create a different type for this section

	skipElem    []string // hard-coded or deprecated elements
	skipAttr    []string // attributes to exclude
	predefined  []string // hard-coded elements exist of the form <name>Elem
	forceRepeat []string // elements to make slices despite DTD
}

var files = []dtd{
	{
		file: "ldmlBCP47",
		root: "LDMLBCP47",
		top:  []string{"ldmlBCP47"},
		skipElem: []string{
			"cldrVersion", // deprecated, not used
		},
	},
	{
		file: "ldmlSupplemental",
		root: "SupplementalData",
		top:  []string{"supplementalData"},
		skipElem: []string{
			"cldrVersion", // deprecated, not used
		},
		forceRepeat: []string{
			"plurals", // data defined in plurals.xml and ordinals.xml
		},
	},
	{
		file: "ldml",
		root: "LDML",
		top: []string{
			"ldml", "collation", "calendar", "timeZoneNames", "localeDisplayNames", "numbers",
		},
		skipElem: []string{
			"cp",       // not used anywhere
			"special",  // not used anywhere
			"fallback", // deprecated, not used
			"alias",    // in Common
			"default",  // in Common
		},
		skipAttr: []string{
			"hiraganaQuarternary", // typo in DTD, correct version included as well
		},
		predefined: []string{"rules"},
	},
}

var comments = map[string]string{
	"ldmlBCP47": `
// LDMLBCP47 holds information on allowable values for various variables in LDML.
`,
	"supplementalData": `
// SupplementalData holds information relevant for internationalization
// and proper use of CLDR, but that is not contained in the locale hierarchy.
`,
	"ldml": `
// LDML is the top-level type for locale-specific data.
`,
	"collation": `
// Collation contains rules that specify a certain sort-order,
// as a tailoring of the root order. 
// The parsed rules are obtained by passing a RuleProcessor to Collation's
// Process method.
`,
	"calendar": `
// Calendar specifies the fields used for formatting and parsing dates and times.
// The month and quarter names are identified numerically, starting at 1.
// The day (of the week) names are identified with short strings, since there is
// no universally-accepted numeric designation.
`,
	"dates": `
// Dates contains information regarding the format and parsing of dates and times.
`,
	"localeDisplayNames": `
// LocaleDisplayNames specifies localized display names for for scripts, languages,
// countries, currencies, and variants.
`,
	"numbers": `
// Numbers supplies information for formatting and parsing numbers and currencies.
`,
}

type element struct {
	name      string // XML element name
	category  string // elements contained by this element
	signature string // category + attrKey*

	attr []*attribute // attributes supported by this element.
	sub  []struct {   // parsed and evaluated sub elements of this element.
		e      *element
		repeat bool // true if the element needs to be a slice
	}

	resolved bool // prevent multiple resolutions of this element.
}

type attribute struct {
	name string
	key  string
	list []string

	tag string // Go tag
}

var (
	reHead  = regexp.MustCompile(` *(\w+) +([\w\-]+)`)
	reAttr  = regexp.MustCompile(` *(\w+) *(?:(\w+)|\(([\w\- \|]+)\)) *(?:#([A-Z]*) *(?:\"([\.\d+])\")?)? *("[\w\-:]*")?`)
	reElem  = regexp.MustCompile(`^ *(EMPTY|ANY|\(.*\)[\*\+\?]?) *$`)
	reToken = regexp.MustCompile(`\w\-`)
)

// builder is used to read in the DTD files from CLDR and generate Go code
// to be used with the encoding/xml package.
type builder struct {
	w       io.Writer
	index   map[string]*element
	elem    []*element
	info    dtd
	version string
}

func makeBuilder(w io.Writer, d dtd) builder {
	return builder{
		w:     w,
		index: make(map[string]*element),
		elem:  []*element{},
		info:  d,
	}
}

// parseDTD parses a DTD file.
func (b *builder) parseDTD(r io.Reader) {
	for d := xml.NewDecoder(r); ; {
		t, err := d.Token()
		if t == nil {
			break
		}
		failOnError(err)
		dir, ok := t.(xml.Directive)
		if !ok {
			continue
		}
		m := reHead.FindSubmatch(dir)
		dir = dir[len(m[0]):]
		ename := string(m[2])
		el, elementFound := b.index[ename]
		switch string(m[1]) {
		case "ELEMENT":
			if elementFound {
				log.Fatal("parseDTD: duplicate entry for element %q", ename)
			}
			m := reElem.FindSubmatch(dir)
			if m == nil {
				log.Fatalf("parseDTD: invalid element %q", string(dir))
			}
			if len(m[0]) != len(dir) {
				log.Fatal("parseDTD: invalid element %q", string(dir), len(dir), len(m[0]), string(m[0]))
			}
			s := string(m[1])
			el = &element{
				name:     ename,
				category: s,
			}
			b.index[ename] = el
		case "ATTLIST":
			if !elementFound {
				log.Fatalf("parseDTD: unknown element %q", ename)
			}
			s := string(dir)
			m := reAttr.FindStringSubmatch(s)
			if m == nil {
				log.Fatal(fmt.Errorf("parseDTD: invalid attribute %q", string(dir)))
			}
			if m[4] == "FIXED" {
				b.version = m[5]
			} else {
				switch m[1] {
				case "draft", "references", "alt", "validSubLocales", "standard" /* in Common */ :
				case "type", "choice":
				default:
					el.attr = append(el.attr, &attribute{
						name: m[1],
						key:  s,
						list: reToken.FindAllString(m[3], -1),
					})
					el.signature = fmt.Sprintf("%s=%s+%s", el.signature, m[1], m[2])
				}
			}
		}
	}
}

var reCat = regexp.MustCompile(`[ ,\|]*(?:(\(|\)|\#?[\w_-]+)([\*\+\?]?))?`)

// resolve takes a parsed element and converts it into structured data
// that can be used to generate the XML code.
func (b *builder) resolve(e *element) {
	if e.resolved {
		return
	}
	b.elem = append(b.elem, e)
	e.resolved = true
	s := e.category
	found := make(map[string]bool)
	sequenceStart := []int{}
	for len(s) > 0 {
		m := reCat.FindStringSubmatch(s)
		if m == nil {
			log.Fatalf("%s: invalid category string %q", e.name, s)
		}
		repeat := m[2] == "*" || m[2] == "+" || in(b.info.forceRepeat, m[1])
		switch m[1] {
		case "":
		case "(":
			sequenceStart = append(sequenceStart, len(e.sub))
		case ")":
			if len(sequenceStart) == 0 {
				log.Fatalf("%s: unmatched closing parenthesis", e.name)
			}
			for i := sequenceStart[len(sequenceStart)-1]; i < len(e.sub); i++ {
				e.sub[i].repeat = e.sub[i].repeat || repeat
			}
			sequenceStart = sequenceStart[:len(sequenceStart)-1]
		default:
			if in(b.info.skipElem, m[1]) {
			} else if sub, ok := b.index[m[1]]; ok {
				if !found[sub.name] {
					e.sub = append(e.sub, struct {
						e      *element
						repeat bool
					}{sub, repeat})
					found[sub.name] = true
					b.resolve(sub)
				}
			} else if m[1] == "#PCDATA" || m[1] == "ANY" {
			} else if m[1] != "EMPTY" {
				log.Fatalf("resolve:%s: element %q not found", e.name, m[1])
			}
		}
		s = s[len(m[0]):]
	}
}

// return true if s is contained in set.
func in(set []string, s string) bool {
	for _, v := range set {
		if v == s {
			return true
		}
	}
	return false
}

var repl = strings.NewReplacer("-", " ", "_", " ")

// title puts the first character or each character following '_' in title case and
// removes all occurrences of '_'.
func title(s string) string {
	return strings.Replace(strings.Title(repl.Replace(s)), " ", "", -1)
}

// writeElem generates Go code for a single element, recursively.
func (b *builder) writeElem(tab int, e *element) {
	p := func(f string, x ...interface{}) {
		f = strings.Replace(f, "\n", "\n"+strings.Repeat("\t", tab), -1)
		fmt.Fprintf(b.w, f, x...)
	}
	if len(e.sub) == 0 && len(e.attr) == 0 {
		p("Common")
		return
	}
	p("struct {")
	tab++
	p("\nCommon")
	for _, attr := range e.attr {
		if !in(b.info.skipAttr, attr.name) {
			p("\n%s string `xml:\"%s,attr\"`", title(attr.name), attr.name)
		}
	}
	for _, sub := range e.sub {
		if in(b.info.predefined, sub.e.name) {
			p("\n%sElem", sub.e.name)
			continue
		}
		if in(b.info.skipElem, sub.e.name) {
			continue
		}
		p("\n%s ", title(sub.e.name))
		if sub.repeat {
			p("[]")
		}
		p("*")
		if in(b.info.top, sub.e.name) {
			p(title(sub.e.name))
		} else {
			b.writeElem(tab, sub.e)
		}
		p(" `xml:\"%s\"`", sub.e.name)
	}
	tab--
	p("\n}")
}

// write generates the Go XML code.
func (b *builder) write() {
	for i, name := range b.info.top {
		e := b.index[name]
		if e != nil {
			fmt.Fprintf(b.w, comments[name])
			name := title(e.name)
			if i == 0 {
				name = b.info.root
			}
			fmt.Fprintf(b.w, "type %s ", name)
			b.writeElem(0, e)
			fmt.Fprint(b.w, "\n")
		}
	}
}
