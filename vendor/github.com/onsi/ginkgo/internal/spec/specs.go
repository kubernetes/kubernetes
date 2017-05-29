package spec

import (
	"math/rand"
	"regexp"
	"sort"
)

type Specs struct {
	specs                []*Spec
	hasProgrammaticFocus bool
	RegexScansFilePath   bool
}

func NewSpecs(specs []*Spec) *Specs {
	return &Specs{
		specs: specs,
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
	for i, j := range permutation {
		shuffledSpecs[i] = e.specs[j]
	}
	e.specs = shuffledSpecs
}

func (e *Specs) ApplyFocus(description string, focusString string, skipString string) {
	if focusString == "" && skipString == "" {
		e.applyProgrammaticFocus()
	} else {
		e.applyRegExpFocusAndSkip(description, focusString, skipString)
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
func (e *Specs) toMatch(description string, spec *Spec) []byte {
	if e.RegexScansFilePath {
		return []byte(
			description + " " +
				spec.ConcatenatedString() + " " +
				spec.subject.CodeLocation().FileName)
	} else {
		return []byte(
			description + " " +
				spec.ConcatenatedString())
	}
}

func (e *Specs) applyRegExpFocusAndSkip(description string, focusString string, skipString string) {
	for _, spec := range e.specs {
		matchesFocus := true
		matchesSkip := false

		toMatch := e.toMatch(description, spec)

		if focusString != "" {
			focusFilter := regexp.MustCompile(focusString)
			matchesFocus = focusFilter.Match([]byte(toMatch))
		}

		if skipString != "" {
			skipFilter := regexp.MustCompile(skipString)
			matchesSkip = skipFilter.Match([]byte(toMatch))
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
	return e.specs[i].ConcatenatedString() < e.specs[j].ConcatenatedString()
}

func (e *Specs) Swap(i, j int) {
	e.specs[i], e.specs[j] = e.specs[j], e.specs[i]
}
