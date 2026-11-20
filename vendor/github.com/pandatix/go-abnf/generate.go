package goabnf

import (
	"errors"
	"maps"
	"math/rand"
	"strings"
	"unicode/utf8"
)

// Generate is an experimental feature that consumes a seed for
// a pseudo-random number generator, used to randomly travel through
// the grammar given a rulename to work on.
// With this, it provides reproducibility, usefull for fuzz crashers.
//
// You can leverage its capacities (maximum repetitions, length
// threshold, etc.) with the functional options.
//
// It is a good capability for testing and fuzzing parsers during
// testing, compliance, fuzzing or optimization.
func (g *Grammar) Generate(seed int64, rulename string, opts ...GenerateOption) ([]byte, error) {
	if err := checkCanGenerateSafely(g, rulename); err != nil {
		return nil, err
	}

	// Use a pseudo-random number generator
	rand := rand.NewSource(seed)

	// Define conditions from options
	options := &genOpts{
		repMax:    16,
		threshold: 256,
	}
	for _, opt := range opts {
		opt.apply(options)
	}

	// Generate actual content (rule exists, checked first)
	rule := GetRule(rulename, g.Rulemap)
	out := []byte{}
	generateAlt(rand, g, &out, rule.Alternation, options)
	return out, nil
}

func generateAlt(rand rand.Source, g *Grammar, out *[]byte, alt Alternation, options *genOpts) {
	// Select any possible path in the alternation
	cnt := alt.Concatenations[int(rand.Int63())%len(alt.Concatenations)]

	// Travel through all the repetitions
	for _, rep := range cnt.Repetitions {
		repmax := rep.Max
		if repmax == inf {
			repmax = options.repMax
		}
		torep := rep.Min + int(rand.Int63())%(repmax-rep.Min+1)
		for i := 0; generateKeepGoing(i, torep, rep.Min, len(*out), options.threshold); i++ {
			switch elem := rep.Element.(type) {
			case ElemRulename:
				rule := GetRule(elem.Name, g.Rulemap)
				generateAlt(rand, g, out, rule.Alternation, options)

			case ElemOption:
				if (rand.Int63() % 2) == 0 {
					generateAlt(rand, g, out, elem.Alternation, options)
				}

			case ElemGroup:
				generateAlt(rand, g, out, elem.Alternation, options)

			case ElemNumVal:
				switch elem.Status {
				case StatRange:
					min, max := numvalToRune(elem.Elems[0], elem.Base), numvalToRune(elem.Elems[1], elem.Base)
					// Generation emits UTF-8 text, so clamp to the representable
					// (Unicode) subset; a range wholly above U+10FFFF (or with
					// max < min) yields no character.
					if max > utf8.MaxRune {
						max = utf8.MaxRune
					}
					if min <= utf8.MaxRune && min <= max {
						r := min + (rune(rand.Int63()&0x0F) % (max - min + 1)) // mask 0x0F to keep the 32-bit part (rune size)
						appendPtr(out, r)
					}

				case StatSeries:
					for _, v := range elem.Elems {
						r := numvalToRune(v, elem.Base)
						// Skip values that are not real characters; they cannot be
						// emitted as UTF-8 text.
						if utf8.ValidRune(r) {
							appendPtr(out, r)
						}
					}
				}

			case ElemCharVal:
				for _, val := range elem.Values {
					if !elem.Sensitive && (int(rand.Int63())%2) == 0 {
						val = runeMax(val)
					}
					appendPtr(out, val)
				}
			}

			// Prose-val is not covered, checked before to avoid generation
		}
	}
}

func appendPtr(slc *[]byte, v ...rune) {
	b := *slc
	b = append(b, []byte(string(v))...)
	*slc = b
}

func generateKeepGoing(i, torep, minrep, lenout, threshold int) bool {
	// If can do no run and reached threshold, skip
	if i == 0 && minrep == 0 && lenout >= threshold {
		return false
	}
	return i < torep
}

// GenerateOption is an option for the *Grammar.Generate method.
type GenerateOption interface {
	apply(*genOpts)
}

type genOpts struct {
	repMax    int
	threshold int
}

type repMaxOption int

// WithRepMax defines the maximum repetition to stop generating at.
func WithRepMax(repMax int) repMaxOption {
	return repMaxOption(repMax)
}

func (opt repMaxOption) apply(opts *genOpts) {
	opts.repMax = int(opt)
}

type thresholdOption int

// WithThreshold defines the length threshold to stop generating at.
func WithThreshold(threshold int) thresholdOption {
	return thresholdOption(threshold)
}

func (opt thresholdOption) apply(opts *genOpts) {
	opts.threshold = int(opt)
}

// checkCanGenerateSafely returns no error if the rule can be generated
// safely i.e. if the rule can exist without infinite recursion, AND if it
// does not contain a prose-val as it could not generate from it.
// Factually, it checks if all involved rules have no path v such that it
// produces a cycle (v:rule-*->rule) AND that this path is mandatory
// (no option, no repetition with a minimum of zero).
func checkCanGenerateSafely(g *Grammar, rulename string) error {
	rule := GetRule(rulename, g.Rulemap)
	if rule == nil {
		return &ErrRuleNotFound{
			Rulename: rulename,
		}
	}
	knownRules := map[string]struct{}{
		rulename: {},
	}
	if err := checkCanGenerateSafelyAlt(g, knownRules, rule.Alternation); err != nil {
		if err, ok := err.(*ErrCyclicRule); ok {
			return err
		}
		return &ErrCyclicRule{
			Rulename: rulename,
		}
	}
	return nil
}

func checkCanGenerateSafelyAlt(g *Grammar, knownRules map[string]struct{}, alt Alternation) error {
	errs := make([]error, len(alt.Concatenations))
	for alti, concat := range alt.Concatenations {
		errs[alti] = checkCanGenerateSafelyConcat(g, knownRules, concat)
	}
	allErrors := true
	for i := 0; i < len(errs) && allErrors; i++ {
		if errs[i] == nil {
			allErrors = false
		}
	}
	if allErrors {
		return errors.Join(errs...)
	}
	return nil
}

func checkCanGenerateSafelyConcat(g *Grammar, knownRules map[string]struct{}, concat Concatenation) error {
	for _, rep := range concat.Repetitions {
		// If the repetition is not mandatory, we can escape so can
		// generate safely.
		if rep.Min == 0 {
			continue
		}

		// Deal with the repetition itself then.
		switch elem := rep.Element.(type) {
		case ElemRulename:
			// Copy rules to only focus on rules that made use come here.
			// If shared with others, the dependency graph can lead to the same rule
			// from another path without it being a cycle, thus must be handled.
			scopeRules := duplicateMap(knownRules)
			for known := range scopeRules {
				if strings.EqualFold(elem.Name, known) {
					return &ErrCyclicRule{
						Rulename: elem.Name,
					}
				}
			}
			rule := GetRule(elem.Name, g.Rulemap)
			scopeRules[elem.Name] = struct{}{}
			if err := checkCanGenerateSafelyAlt(g, scopeRules, rule.Alternation); err != nil {
				if err, ok := err.(*ErrCyclicRule); ok {
					return err
				}
				return &ErrCyclicRule{
					Rulename: elem.Name,
				}
			}

		case ElemGroup:
			if err := checkCanGenerateSafelyAlt(g, knownRules, elem.Alternation); err != nil {
				return err
			}

		case ElemProseVal:
			return ErrHandlingProseVal

			// Other types are not considered for the following reasons:
			// - option: equivalent to rep.min==0, escapable path even if could be cyclic
			// - num-val, char-val: termination paths, can't be cyclic
		}
	}
	return nil
}

func duplicateMap[T comparable, V any](m map[T]V) map[T]V {
	n := make(map[T]V, len(m))
	maps.Copy(n, m)
	return n
}
