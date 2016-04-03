package spec

import (
	"math/rand"
	"regexp"
	"sort"
)

type Specs struct {
	specs                 []*Spec
	numberOfOriginalSpecs int
	hasProgrammaticFocus  bool
}

func NewSpecs(specs []*Spec) *Specs {
	return &Specs{
		specs: specs,
		numberOfOriginalSpecs: len(specs),
	}
}

func (e *Specs) Specs() []*Spec {
	return e.specs
}

func (e *Specs) NumberOfOriginalSpecs() int {
	return e.numberOfOriginalSpecs
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
		e.applyRegExpFocus(description, focusString, skipString)
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

func (e *Specs) applyRegExpFocus(description string, focusString string, skipString string) {
	for _, spec := range e.specs {
		matchesFocus := true
		matchesSkip := false

		toMatch := []byte(description + " " + spec.ConcatenatedString())

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

func (e *Specs) TrimForParallelization(total int, node int) {
	startIndex, count := ParallelizedIndexRange(len(e.specs), total, node)
	if count == 0 {
		e.specs = make([]*Spec, 0)
	} else {
		e.specs = e.specs[startIndex : startIndex+count]
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
