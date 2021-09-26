// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pipeline provides tools for creating translation pipelines.
//
// NOTE: UNDER DEVELOPMENT. API MAY CHANGE.
package pipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/build"
	"go/parser"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"text/template"
	"unicode"

	"golang.org/x/text/internal"
	"golang.org/x/text/language"
	"golang.org/x/text/runes"
	"golang.org/x/tools/go/loader"
)

const (
	extractFile  = "extracted.gotext.json"
	outFile      = "out.gotext.json"
	gotextSuffix = "gotext.json"
)

// Config contains configuration for the translation pipeline.
type Config struct {
	// Supported indicates the languages for which data should be generated.
	// The default is to support all locales for which there are matching
	// translation files.
	Supported []language.Tag

	// --- Extraction

	SourceLanguage language.Tag

	Packages []string

	// --- File structure

	// Dir is the root dir for all operations.
	Dir string

	// TranslationsPattern is a regular expression to match incoming translation
	// files. These files may appear in any directory rooted at Dir.
	// language for the translation files is determined as follows:
	//   1. From the Language field in the file.
	//   2. If not present, from a valid language tag in the filename, separated
	//      by dots (e.g. "en-US.json" or "incoming.pt_PT.xmb").
	//   3. If not present, from a the closest subdirectory in which the file
	//      is contained that parses as a valid language tag.
	TranslationsPattern string

	// OutPattern defines the location for translation files for a certain
	// language. The default is "{{.Dir}}/{{.Language}}/out.{{.Ext}}"
	OutPattern string

	// Format defines the file format for generated translation files.
	// The default is XMB. Alternatives are GetText, XLIFF, L20n, GoText.
	Format string

	Ext string

	// TODO:
	// Actions are additional actions to be performed after the initial extract
	// and merge.
	// Actions []struct {
	// 	Name    string
	// 	Options map[string]string
	// }

	// --- Generation

	// GenFile may be in a different package. It is not defined, it will
	// be written to stdout.
	GenFile string

	// GenPackage is the package or relative path into which to generate the
	// file. If not specified it is relative to the current directory.
	GenPackage string

	// DeclareVar defines a variable to which to assing the generated Catalog.
	DeclareVar string

	// SetDefault determines whether to assign the generated Catalog to
	// message.DefaultCatalog. The default for this is true if DeclareVar is
	// not defined, false otherwise.
	SetDefault bool

	// TODO:
	// - Printf-style configuration
	// - Template-style configuration
	// - Extraction options
	// - Rewrite options
	// - Generation options
}

// Operations:
// - extract:       get the strings
// - disambiguate:  find messages with the same key, but possible different meaning.
// - create out:    create a list of messages that need translations
// - load trans:    load the list of current translations
// - merge:         assign list of translations as done
// - (action)expand:    analyze features and create example sentences for each version.
// - (action)googletrans:   pre-populate messages with automatic translations.
// - (action)export:    send out messages somewhere non-standard
// - (action)import:    load messages from somewhere non-standard
// - vet program:   don't pass "foo" + var + "bar" strings. Not using funcs for translated strings.
// - vet trans:     coverage: all translations/ all features.
// - generate:      generate Go code

// State holds all accumulated information on translations during processing.
type State struct {
	Config Config

	Package string
	program *loader.Program

	Extracted Messages `json:"messages"`

	// Messages includes all messages for which there need to be translations.
	// Duplicates may be eliminated. Generation will be done from these messages
	// (usually after merging).
	Messages []Messages

	// Translations are incoming translations for the application messages.
	Translations []Messages
}

func (s *State) dir() string {
	if d := s.Config.Dir; d != "" {
		return d
	}
	return "./locales"
}

func outPattern(s *State) (string, error) {
	c := s.Config
	pat := c.OutPattern
	if pat == "" {
		pat = "{{.Dir}}/{{.Language}}/out.{{.Ext}}"
	}

	ext := c.Ext
	if ext == "" {
		ext = c.Format
	}
	if ext == "" {
		ext = gotextSuffix
	}
	t, err := template.New("").Parse(pat)
	if err != nil {
		return "", wrap(err, "error parsing template")
	}
	buf := bytes.Buffer{}
	err = t.Execute(&buf, map[string]string{
		"Dir":      s.dir(),
		"Language": "%s",
		"Ext":      ext,
	})
	return filepath.FromSlash(buf.String()), wrap(err, "incorrect OutPattern")
}

var transRE = regexp.MustCompile(`.*\.` + gotextSuffix)

// Import loads existing translation files.
func (s *State) Import() error {
	outPattern, err := outPattern(s)
	if err != nil {
		return err
	}
	re := transRE
	if pat := s.Config.TranslationsPattern; pat != "" {
		if re, err = regexp.Compile(pat); err != nil {
			return wrapf(err, "error parsing regexp %q", s.Config.TranslationsPattern)
		}
	}
	x := importer{s, outPattern, re}
	return x.walkImport(s.dir(), s.Config.SourceLanguage)
}

type importer struct {
	state      *State
	outPattern string
	transFile  *regexp.Regexp
}

func (i *importer) walkImport(path string, tag language.Tag) error {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil
	}
	for _, f := range files {
		name := f.Name()
		tag := tag
		if f.IsDir() {
			if t, err := language.Parse(name); err == nil {
				tag = t
			}
			// We ignore errors
			if err := i.walkImport(filepath.Join(path, name), tag); err != nil {
				return err
			}
			continue
		}
		for _, l := range strings.Split(name, ".") {
			if t, err := language.Parse(l); err == nil {
				tag = t
			}
		}
		file := filepath.Join(path, name)
		// TODO: Should we skip files that match output files?
		if fmt.Sprintf(i.outPattern, tag) == file {
			continue
		}
		// TODO: handle different file formats.
		if !i.transFile.MatchString(name) {
			continue
		}
		b, err := ioutil.ReadFile(file)
		if err != nil {
			return wrap(err, "read file failed")
		}
		var translations Messages
		if err := json.Unmarshal(b, &translations); err != nil {
			return wrap(err, "parsing translation file failed")
		}
		i.state.Translations = append(i.state.Translations, translations)
	}
	return nil
}

// Merge merges the extracted messages with the existing translations.
func (s *State) Merge() error {
	if s.Messages != nil {
		panic("already merged")
	}
	// Create an index for each unique message.
	// Duplicates are okay as long as the substitution arguments are okay as
	// well.
	// Top-level messages are okay to appear in multiple substitution points.

	// Collect key equivalence.
	msgs := []*Message{}
	keyToIDs := map[string]*Message{}
	for _, m := range s.Extracted.Messages {
		m := m
		if prev, ok := keyToIDs[m.Key]; ok {
			if err := checkEquivalence(&m, prev); err != nil {
				warnf("Key %q matches conflicting messages: %v and %v", m.Key, prev.ID, m.ID)
				// TODO: track enough information so that the rewriter can
				// suggest/disambiguate messages.
			}
			// TODO: add position to message.
			continue
		}
		i := len(msgs)
		msgs = append(msgs, &m)
		keyToIDs[m.Key] = msgs[i]
	}

	// Messages with different keys may still refer to the same translated
	// message (e.g. different whitespace). Filter these.
	idMap := map[string]bool{}
	filtered := []*Message{}
	for _, m := range msgs {
		found := false
		for _, id := range m.ID {
			found = found || idMap[id]
		}
		if !found {
			filtered = append(filtered, m)
		}
		for _, id := range m.ID {
			idMap[id] = true
		}
	}

	// Build index of translations.
	translations := map[language.Tag]map[string]Message{}
	languages := append([]language.Tag{}, s.Config.Supported...)

	for _, t := range s.Translations {
		tag := t.Language
		if _, ok := translations[tag]; !ok {
			translations[tag] = map[string]Message{}
			languages = append(languages, tag)
		}
		for _, m := range t.Messages {
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
	languages = internal.UniqueTags(languages)

	for _, tag := range languages {
		ms := Messages{Language: tag}
		for _, orig := range filtered {
			m := *orig
			m.Key = ""
			m.Position = ""

			for _, id := range m.ID {
				if t, ok := translations[tag][id]; ok {
					m.Translation = t.Translation
					if t.TranslatorComment != "" {
						m.TranslatorComment = t.TranslatorComment
						m.Fuzzy = t.Fuzzy
					}
					break
				}
			}
			if tag == s.Config.SourceLanguage && m.Translation.IsEmpty() {
				m.Translation = m.Message
				if m.TranslatorComment == "" {
					m.TranslatorComment = "Copied from source."
					m.Fuzzy = true
				}
			}
			// TODO: if translation is empty: pre-expand based on available
			// linguistic features. This may also be done as a plugin.
			ms.Messages = append(ms.Messages, m)
		}
		s.Messages = append(s.Messages, ms)
	}
	return nil
}

// Export writes out the messages to translation out files.
func (s *State) Export() error {
	path, err := outPattern(s)
	if err != nil {
		return wrap(err, "export failed")
	}
	for _, out := range s.Messages {
		// TODO: inject translations from existing files to avoid retranslation.
		data, err := json.MarshalIndent(out, "", "    ")
		if err != nil {
			return wrap(err, "JSON marshal failed")
		}
		file := fmt.Sprintf(path, out.Language)
		if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
			return wrap(err, "dir create failed")
		}
		if err := ioutil.WriteFile(file, data, 0644); err != nil {
			return wrap(err, "write failed")
		}
	}
	return nil
}

var (
	ws    = runes.In(unicode.White_Space).Contains
	notWS = runes.NotIn(unicode.White_Space).Contains
)

func trimWS(s string) (trimmed, leadWS, trailWS string) {
	trimmed = strings.TrimRightFunc(s, ws)
	trailWS = s[len(trimmed):]
	if i := strings.IndexFunc(trimmed, notWS); i > 0 {
		leadWS = trimmed[:i]
		trimmed = trimmed[i:]
	}
	return trimmed, leadWS, trailWS
}

// NOTE: The command line tool already prefixes with "gotext:".
var (
	wrap = func(err error, msg string) error {
		if err == nil {
			return nil
		}
		return fmt.Errorf("%s: %v", msg, err)
	}
	wrapf = func(err error, msg string, args ...interface{}) error {
		if err == nil {
			return nil
		}
		return wrap(err, fmt.Sprintf(msg, args...))
	}
	errorf = fmt.Errorf
)

func warnf(format string, args ...interface{}) {
	// TODO: don't log.
	log.Printf(format, args...)
}

func loadPackages(conf *loader.Config, args []string) (*loader.Program, error) {
	if len(args) == 0 {
		args = []string{"."}
	}

	conf.Build = &build.Default
	conf.ParserMode = parser.ParseComments

	// Use the initial packages from the command line.
	args, err := conf.FromArgs(args, false)
	if err != nil {
		return nil, wrap(err, "loading packages failed")
	}

	// Load, parse and type-check the whole program.
	return conf.Load()
}
