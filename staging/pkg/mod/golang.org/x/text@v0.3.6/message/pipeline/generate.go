// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pipeline

import (
	"fmt"
	"go/build"
	"io"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/template"

	"golang.org/x/text/collate"
	"golang.org/x/text/feature/plural"
	"golang.org/x/text/internal"
	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
	"golang.org/x/tools/go/loader"
)

var transRe = regexp.MustCompile(`messages\.(.*)\.json`)

// Generate writes a Go file that defines a Catalog with translated messages.
// Translations are retrieved from s.Messages, not s.Translations, so it
// is assumed Merge has been called.
func (s *State) Generate() error {
	path := s.Config.GenPackage
	if path == "" {
		path = "."
	}
	isDir := path[0] == '.'
	prog, err := loadPackages(&loader.Config{}, []string{path})
	if err != nil {
		return wrap(err, "could not load package")
	}
	pkgs := prog.InitialPackages()
	if len(pkgs) != 1 {
		return errorf("more than one package selected: %v", pkgs)
	}
	pkg := pkgs[0].Pkg.Name()

	cw, err := s.generate()
	if err != nil {
		return err
	}
	if !isDir {
		gopath := filepath.SplitList(build.Default.GOPATH)[0]
		path = filepath.Join(gopath, "src", filepath.FromSlash(pkgs[0].Pkg.Path()))
	}
	if filepath.IsAbs(s.Config.GenFile) {
		path = s.Config.GenFile
	} else {
		path = filepath.Join(path, s.Config.GenFile)
	}
	cw.WriteGoFile(path, pkg) // TODO: WriteGoFile should return error.
	return err
}

// WriteGen writes a Go file with the given package name to w that defines a
// Catalog with translated messages. Translations are retrieved from s.Messages,
// not s.Translations, so it is assumed Merge has been called.
func (s *State) WriteGen(w io.Writer, pkg string) error {
	cw, err := s.generate()
	if err != nil {
		return err
	}
	_, err = cw.WriteGo(w, pkg, "")
	return err
}

// Generate is deprecated; use (*State).Generate().
func Generate(w io.Writer, pkg string, extracted *Messages, trans ...Messages) (n int, err error) {
	s := State{
		Extracted:    *extracted,
		Translations: trans,
	}
	cw, err := s.generate()
	if err != nil {
		return 0, err
	}
	return cw.WriteGo(w, pkg, "")
}

func (s *State) generate() (*gen.CodeWriter, error) {
	// Build up index of translations and original messages.
	translations := map[language.Tag]map[string]Message{}
	languages := []language.Tag{}
	usedKeys := map[string]int{}

	for _, loc := range s.Messages {
		tag := loc.Language
		if _, ok := translations[tag]; !ok {
			translations[tag] = map[string]Message{}
			languages = append(languages, tag)
		}
		for _, m := range loc.Messages {
			if !m.Translation.IsEmpty() {
				for _, id := range m.ID {
					if _, ok := translations[tag][id]; ok {
						warnf("Duplicate translation in locale %q for message %q", tag, id)
					}
					translations[tag][id] = m
				}
			}
		}
	}

	// Verify completeness and register keys.
	internal.SortTags(languages)

	langVars := []string{}
	for _, tag := range languages {
		langVars = append(langVars, strings.Replace(tag.String(), "-", "_", -1))
		dict := translations[tag]
		for _, msg := range s.Extracted.Messages {
			for _, id := range msg.ID {
				if trans, ok := dict[id]; ok && !trans.Translation.IsEmpty() {
					if _, ok := usedKeys[msg.Key]; !ok {
						usedKeys[msg.Key] = len(usedKeys)
					}
					break
				}
				// TODO: log missing entry.
				warnf("%s: Missing entry for %q.", tag, id)
			}
		}
	}

	cw := gen.NewCodeWriter()

	x := &struct {
		Fallback  language.Tag
		Languages []string
	}{
		Fallback:  s.Extracted.Language,
		Languages: langVars,
	}

	if err := lookup.Execute(cw, x); err != nil {
		return nil, wrap(err, "error")
	}

	keyToIndex := []string{}
	for k := range usedKeys {
		keyToIndex = append(keyToIndex, k)
	}
	sort.Strings(keyToIndex)
	fmt.Fprint(cw, "var messageKeyToIndex = map[string]int{\n")
	for _, k := range keyToIndex {
		fmt.Fprintf(cw, "%q: %d,\n", k, usedKeys[k])
	}
	fmt.Fprint(cw, "}\n\n")

	for i, tag := range languages {
		dict := translations[tag]
		a := make([]string, len(usedKeys))
		for _, msg := range s.Extracted.Messages {
			for _, id := range msg.ID {
				if trans, ok := dict[id]; ok && !trans.Translation.IsEmpty() {
					m, err := assemble(&msg, &trans.Translation)
					if err != nil {
						return nil, wrap(err, "error")
					}
					_, leadWS, trailWS := trimWS(msg.Key)
					if leadWS != "" || trailWS != "" {
						m = catmsg.Affix{
							Message: m,
							Prefix:  leadWS,
							Suffix:  trailWS,
						}
					}
					// TODO: support macros.
					data, err := catmsg.Compile(tag, nil, m)
					if err != nil {
						return nil, wrap(err, "error")
					}
					key := usedKeys[msg.Key]
					if d := a[key]; d != "" && d != data {
						warnf("Duplicate non-consistent translation for key %q, picking the one for message %q", msg.Key, id)
					}
					a[key] = string(data)
					break
				}
			}
		}
		index := []uint32{0}
		p := 0
		for _, s := range a {
			p += len(s)
			index = append(index, uint32(p))
		}

		cw.WriteVar(langVars[i]+"Index", index)
		cw.WriteConst(langVars[i]+"Data", strings.Join(a, ""))
	}
	return cw, nil
}

func assemble(m *Message, t *Text) (msg catmsg.Message, err error) {
	keys := []string{}
	for k := range t.Var {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var a []catmsg.Message
	for _, k := range keys {
		t := t.Var[k]
		m, err := assemble(m, &t)
		if err != nil {
			return nil, err
		}
		a = append(a, &catmsg.Var{Name: k, Message: m})
	}
	if t.Select != nil {
		s, err := assembleSelect(m, t.Select)
		if err != nil {
			return nil, err
		}
		a = append(a, s)
	}
	if t.Msg != "" {
		sub, err := m.Substitute(t.Msg)
		if err != nil {
			return nil, err
		}
		a = append(a, catmsg.String(sub))
	}
	switch len(a) {
	case 0:
		return nil, errorf("generate: empty message")
	case 1:
		return a[0], nil
	default:
		return catmsg.FirstOf(a), nil

	}
}

func assembleSelect(m *Message, s *Select) (msg catmsg.Message, err error) {
	cases := []string{}
	for c := range s.Cases {
		cases = append(cases, c)
	}
	sortCases(cases)

	caseMsg := []interface{}{}
	for _, c := range cases {
		cm := s.Cases[c]
		m, err := assemble(m, &cm)
		if err != nil {
			return nil, err
		}
		caseMsg = append(caseMsg, c, m)
	}

	ph := m.Placeholder(s.Arg)

	switch s.Feature {
	case "plural":
		// TODO: only printf-style selects are supported as of yet.
		return plural.Selectf(ph.ArgNum, ph.String, caseMsg...), nil
	}
	return nil, errorf("unknown feature type %q", s.Feature)
}

func sortCases(cases []string) {
	// TODO: implement full interface.
	sort.Slice(cases, func(i, j int) bool {
		switch {
		case cases[i] != "other" && cases[j] == "other":
			return true
		case cases[i] == "other" && cases[j] != "other":
			return false
		}
		// the following code relies on '<' < '=' < any letter.
		return cmpNumeric(cases[i], cases[j]) == -1
	})
}

var cmpNumeric = collate.New(language.Und, collate.Numeric).CompareString

var lookup = template.Must(template.New("gen").Parse(`
import (
	"golang.org/x/text/language"
	"golang.org/x/text/message"
	"golang.org/x/text/message/catalog"
)

type dictionary struct {
	index []uint32
	data  string
}

func (d *dictionary) Lookup(key string) (data string, ok bool) {
	p, ok := messageKeyToIndex[key]
	if !ok {
		return "", false
	}
	start, end := d.index[p], d.index[p+1]
	if start == end {
		return "", false
	}
	return d.data[start:end], true
}

func init() {
	dict := map[string]catalog.Dictionary{
		{{range .Languages}}"{{.}}": &dictionary{index: {{.}}Index, data: {{.}}Data },
		{{end}}
	}
	fallback := language.MustParse("{{.Fallback}}")
	cat, err := catalog.NewFromMap(dict, catalog.Fallback(fallback))
	if err != nil {
		panic(err)
	}
	message.DefaultCatalog = cat
}

`))
