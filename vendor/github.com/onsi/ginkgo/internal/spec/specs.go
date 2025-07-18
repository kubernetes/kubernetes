package spec

import (
	"math/rand"
	"regexp"
	"sort"
	"strings"
)

type Specs struct {
	specs []*Spec
	names []string

	hasProgrammaticFocus bool
	RegexScansFilePath   bool
}

func NewSpecs(specs []*Spec) *Specs {
	names := make([]string, len(specs))
	for i, spec := range specs {
		names[i] = spec.ConcatenatedString()
	}
	return &Specs{
		specs: specs,
		names: names,
	}
}

func (e *Specs) Specs() []*Spec {
	return e.specs
}

func (e *Specs) HasProgrammaticFocus() bool {
	return e.hasProgrammaticFocus
}

func (e *Specs) Shuffle(r *rand.Rand) {
	sort.Sort(e)
	permutation := r.Perm(len(e.specs))
	shuffledSpecs := make([]*Spec, len(e.specs))
	names := make([]string, len(e.specs))
	for i, j := range permutation {
		shuffledSpecs[i] = e.specs[j]
		names[i] = e.names[j]
	}
	e.specs = shuffledSpecs
	e.names = names
}

func (e *Specs) ApplyFocus(description string, focus, skip []string) {
	if len(focus)+len(skip) == 0 {
		e.applyProgrammaticFocus()
	} else {
		e.applyRegExpFocusAndSkip(description, focus, skip)
	}
}

func (e *Specs) applyProgrammaticFocus() {
	e.hasProgrammaticFocus = false
	for _, spec := range e.specs {
		if spec.Focused() && !spec.Pending() {
			e.hasProgrammaticFocus = true
			break
		}
	}

	if e.hasProgrammaticFocus {
		for _, spec := range e.specs {
			if !spec.Focused() {
				spec.Skip()
			}
		}
	}
}

// toMatch returns a byte[] to be used by regex matchers.  When adding new behaviours to the matching function,
// this is the place which we append to.
func (e *Specs) toMatch(description string, i int) []byte {
	if i > len(e.names) {
		return nil
	}
	if e.RegexScansFilePath {
		return []byte(
			description + " " +
				e.names[i] + " " +
				e.specs[i].subject.CodeLocation().FileName)
	} else {
		return []byte(
			description + " " +
				e.names[i])
	}
}

func (e *Specs) applyRegExpFocusAndSkip(description string, focus, skip []string) {
	var focusFilter, skipFilter *regexp.Regexp
	if len(focus) > 0 {
		focusFilter = regexp.MustCompile(strings.Join(focus, "|"))
	}
	if len(skip) > 0 {
		skipFilter = regexp.MustCompile(strings.Join(skip, "|"))
	}

	for i, spec := range e.specs {
		matchesFocus := true
		matchesSkip := false

		toMatch := e.toMatch(description, i)

		if focusFilter != nil {
			matchesFocus = focusFilter.Match(toMatch)
		}

		if skipFilter != nil {
			matchesSkip = skipFilter.Match(toMatch)
		}

		if !matchesFocus || matchesSkip {
			spec.Skip()
		}
	}
}

func (e *Specs) SkipMeasurements() {
	for _, spec := range e.specs {
		if spec.IsMeasurement() {
			spec.Skip()
		}
	}
}

//sort.Interface

func (e *Specs) Len() int {
	return len(e.specs)
}

func (e *Specs) Less(i, j int) bool {
	return e.names[i] < e.names[j]
}

func (e *Specs) Swap(i, j int) {
	e.names[i], e.names[j] = e.names[j], e.names[i]
	e.specs[i], e.specs[j] = e.specs[j], e.specs[i]
}
