package goabnf

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Grammar represents an ABNF grammar as defined by RFC 5234.
// It is constituted of a set of rules with a unique name.
type Grammar struct {
	Rulemap map[string]*Rule
}

// IsValid checks there exist at least a path that completly consumes
// input, hence is valid given this grammar and especially one of its
// rule.
func (g *Grammar) IsValid(rulename string, input []byte) (bool, error) {
	rule := GetRule(rulename, g.Rulemap)
	if rule == nil {
		return false, &ErrRuleNotFound{Rulename: rulename}
	}
	// Validity only needs to know whether some derivation consumes the whole
	// input, never the (possibly exponentially many) derivations themselves.
	// We therefore propagate the set of reachable end-positions per
	// (element, index) instead of enumerating paths, which keeps this
	// polynomial. Left-recursive rules are resolved by seed-growing rather than
	// refused; see recognize.go and leftrec.go.
	r := &recognizer{
		g:          g,
		input:      input,
		memo:       map[string]map[int]bool{},
		inProgress: map[string]bool{},
		leftRec:    g.leftRecursiveSCCs(),
		growing:    map[string]map[int]bool{},
	}
	ends := r.reachElem(ElemRulename{Name: rulename}, 0)
	return ends[len(input)], nil
}

// String returns the representation of the grammar that is valid
// according to the ABNF specifications/RFCs.
// This notably imply the use of CRLF instead of LF, and does not
// preserve the initial order nor pretty print it.
func (g *Grammar) String() string {
	str := ""
	for _, rule := range g.Rulemap {
		str += rule.String() + "\r\n"
	}
	return str
}

// PrettyPrint returns a prettified string that represents the grammar.
func (g *Grammar) PrettyPrint() string {
	// Determine maximum rulename length
	rulenameLength := 0
	for rulename := range g.Rulemap {
		if len(rulename) > rulenameLength {
			rulenameLength = len(rulename)
		}
	}

	// Construct output
	out := ""
	for rulename, rl := range g.Rulemap {
		spaces := ""
		for i := 0; i < rulenameLength-len(rulename); i++ {
			spaces += " "
		}

		out += fmt.Sprintf("%s%s = %s\r\n", rulename, spaces, rl.Alternation)
	}
	return out
}

// ParseABNF is a helper facilitating the call to Parse using the
// pre-computed ABNF grammar and evaluated the resulting to produce
// a ready-to-use grammar.
func ParseABNF(input []byte, opts ...ABNFOption) (*Grammar, error) {
	o := process(opts...)

	// Parse the ABNF source using the ABNF meta-grammar with the GLL engine.
	f, err := Parse(input, ABNF, "rulelist")
	if err != nil {
		return nil, err
	}
	if !f.Valid() {
		return nil, f.ParseError()
	}
	if f.Ambiguous() {
		return nil, &ErrMultipleSolutionsFound{}
	}

	g, err := evalForest(input, f, o)
	if err != nil {
		return nil, err
	}

	if o.validate {
		if err := SemvalABNF(g); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Parse parses input against grammar starting at rootRulename using the GLL
// engine, returning the shared packed parse forest. Unlike the previous
// backtracking parser it accepts ANY grammar -- left, right and mutual
// recursion and ambiguity included -- in worst-case cubic time, so it never
// stack-overflows on left recursion nor blows up exponentially. Inspect the
// result with the Forest methods (Valid, Ambiguous, NumTrees, Tree).
func Parse(input []byte, grammar *Grammar, rootRulename string) (*Forest, error) {
	return ParseForest(input, grammar, rootRulename)
}

// EvaluateABNF evaluates a parse forest -- produced by parsing an ABNF source
// against the ABNF meta-grammar -- into a ready-to-use *Grammar.
func EvaluateABNF(input []byte, f *Forest, opts ...ABNFOption) (*Grammar, error) {
	return evalForest(input, f, process(opts...))
}

func evalForest(input []byte, f *Forest, o *abnfOptions) (*Grammar, error) {
	tree := f.Tree()
	if tree == nil {
		return nil, ErrNoSolutionFound
	}
	ev := &feval{input: input, o: o}
	return ev.rulelist(tree)
}

// feval evaluates the rule-keyed ParseTree of an ABNF source into a *Grammar.
// It dispatches on rule names and reads byte spans, so it is decoupled from the
// exact (scaffolding-collapsed) shape of the tree.
type feval struct {
	input []byte
	o     *abnfOptions
}

func ptChildren(t *ParseTree, name string) []*ParseTree {
	var out []*ParseTree
	for _, c := range t.Children {
		if c.Rule == name {
			out = append(out, c)
		}
	}
	return out
}

func ptFirst(t *ParseTree, names ...string) *ParseTree {
	for _, c := range t.Children {
		if slices.Contains(names, c.Rule) {
			return c
		}
	}
	return nil
}

func (e *feval) span(t *ParseTree) string { return string(e.input[t.Start:t.End]) }

func (e *feval) rulelist(t *ParseTree) (*Grammar, error) {
	mp := map[string]*Rule{}
	for _, rc := range ptChildren(t, "rule") {
		rl, definedAs, err := e.rule(rc)
		if err != nil {
			return nil, err
		}
		switch definedAs {
		case "=":
			isCoreRule := GetRule(rl.Name, nil) != nil
			if isCoreRule {
				if !e.o.redefineCore {
					return nil, &ErrCoreRuleModify{CoreRulename: rl.Name}
				}
			} else if GetRule(rl.Name, mp) != nil {
				return nil, &ErrDuplicatedRule{Rulename: rl.Name}
			}
			mp[rl.Name] = rl
		case "=/":
			isCoreRule := GetRule(rl.Name, nil) != nil
			if !e.o.redefineCore && isCoreRule {
				return nil, &ErrCoreRuleModify{CoreRulename: rl.Name}
			}
			rule := GetRule(rl.Name, mp)
			if rule == nil {
				return nil, &ErrRuleNotFound{Rulename: rl.Name}
			}
			rule.Alternation.Concatenations = append(rule.Alternation.Concatenations, rl.Alternation.Concatenations...)
			mp[rule.Name] = rule
		}
	}
	return &Grammar{Rulemap: mp}, nil
}

func (e *feval) rule(t *ParseTree) (*Rule, string, error) {
	nameNode := ptFirst(t, "rulename")
	defNode := ptFirst(t, "defined-as")
	elemsNode := ptFirst(t, "elements")
	if nameNode == nil || defNode == nil || elemsNode == nil {
		return nil, "", ErrNoSolutionFound
	}
	definedAs := "="
	if strings.Contains(e.span(defNode), "=/") {
		definedAs = "=/"
	}
	altNode := ptFirst(elemsNode, "alternation")
	if altNode == nil {
		return nil, "", ErrNoSolutionFound
	}
	alt, err := e.alternation(altNode)
	if err != nil {
		return nil, "", err
	}
	return &Rule{Name: e.span(nameNode), Alternation: alt}, definedAs, nil
}

func (e *feval) alternation(t *ParseTree) (Alternation, error) {
	var cs []Concatenation
	for _, cc := range ptChildren(t, "concatenation") {
		c, err := e.concatenation(cc)
		if err != nil {
			return Alternation{}, err
		}
		cs = append(cs, c)
	}
	return Alternation{Concatenations: cs}, nil
}

func (e *feval) concatenation(t *ParseTree) (Concatenation, error) {
	var rs []Repetition
	for _, rc := range ptChildren(t, "repetition") {
		r, err := e.repetition(rc)
		if err != nil {
			return Concatenation{}, err
		}
		rs = append(rs, r)
	}
	return Concatenation{Repetitions: rs}, nil
}

func (e *feval) repetition(t *ParseTree) (Repetition, error) {
	min, max := 1, 1
	if rep := ptFirst(t, "repeat"); rep != nil {
		min, max = e.parseRepeat(rep)
	}
	elNode := ptFirst(t, "element")
	if elNode == nil {
		return Repetition{}, ErrNoSolutionFound
	}
	elem, err := e.element(elNode)
	if err != nil {
		return Repetition{}, err
	}
	return Repetition{Min: min, Max: max, Element: elem}, nil
}

func (e *feval) parseRepeat(t *ParseTree) (int, int) {
	s := e.span(t)
	before, after, ok := strings.Cut(s, "*")
	if !ok {
		d, _ := strconv.Atoi(s)
		return d, d
	}
	min, max := 0, inf
	if pre := before; pre != "" {
		min, _ = strconv.Atoi(pre)
	}
	if post := after; post != "" {
		max, _ = strconv.Atoi(post)
	}
	return min, max
}

func (e *feval) element(t *ParseTree) (ElemItf, error) {
	c := ptFirst(t, "rulename", "group", "option", "char-val", "num-val", "prose-val")
	if c == nil {
		return nil, ErrNoSolutionFound
	}
	switch c.Rule {
	case "rulename":
		return ElemRulename{Name: e.span(c)}, nil
	case "group":
		alt, err := e.alternation(ptFirst(c, "alternation"))
		if err != nil {
			return nil, err
		}
		return ElemGroup{Alternation: alt}, nil
	case "option":
		alt, err := e.alternation(ptFirst(c, "alternation"))
		if err != nil {
			return nil, err
		}
		return ElemOption{Alternation: alt}, nil
	case "char-val":
		return e.charVal(c), nil
	case "num-val":
		return e.numVal(c), nil
	case "prose-val":
		return e.proseVal(c), nil
	}
	return nil, ErrNoSolutionFound
}

func (e *feval) charVal(t *ParseTree) ElemCharVal {
	s := e.span(t)
	sensitive := false
	if len(s) >= 2 && s[0] == '%' {
		if s[1] == 's' || s[1] == 'S' {
			sensitive = true
		}
		s = s[2:]
	}
	first := strings.IndexByte(s, '"')
	last := strings.LastIndexByte(s, '"')
	value := []rune{}
	if first >= 0 && last > first {
		value = []rune(s[first+1 : last])
	}
	return ElemCharVal{Sensitive: sensitive, Values: value}
}

func (e *feval) numVal(t *ParseTree) ElemNumVal {
	s := e.span(t)
	base := "d"
	rest := ""
	if len(s) >= 2 && s[0] == '%' {
		switch s[1] {
		case 'b', 'B':
			base = "b"
		case 'd', 'D':
			base = "d"
		case 'x', 'X':
			base = "x"
		}
		rest = s[2:]
	}
	status := StatSeries
	var elems []string
	if strings.ContainsRune(rest, '-') {
		status = StatRange
		elems = strings.Split(rest, "-")
	} else {
		elems = strings.Split(rest, ".")
	}
	// A num-val outside the Unicode range is still a well-formed numeric
	// encoding (legitimate for non-textual / on-wire grammars), so it is kept
	// as-is here. Strict semantic validation (SemvalABNF) rejects it for textual
	// use; consumers that match against decoded runes treat an unrepresentable
	// value as non-matching rather than panicking.
	return ElemNumVal{Base: base, Status: status, Elems: elems}
}

func (e *feval) proseVal(t *ParseTree) ElemProseVal {
	values := []string{}
	for i := t.Start + 1; i < t.End-1; i++ {
		values = append(values, string(e.input[i]))
	}
	return ElemProseVal{values: values}
}

func sensequal(target, actual rune, sensitive bool) bool {
	if !sensitive {
		target, actual = runeMin(target), runeMin(actual)
	}
	return target == actual
}

func runeMin(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r - 'A' + 'a'
	}
	return r
}

func runeMax(r rune) rune {
	if r >= 'a' && r <= 'z' {
		return r - 'a' + 'A'
	}
	return r
}

// SemvalABNF proceed to semantic validations of an ABNF grammar.
// It currently support the following checks:
// - for all rules, its dependencies (rules) exist
// - for repetition, min <= max
// - for num-val, that the value fits in 7-bits (US-ASCII encoded)
// To update this list, please open an issue.
func SemvalABNF(g *Grammar) error {
	// Check all dependencies exist
	for _, rule := range g.Rulemap {
		deps := getDependencies(rule.Alternation)
		for _, dep := range deps {
			r := GetRule(dep, g.Rulemap)
			if r == nil {
				return &ErrDependencyNotFound{
					Rulename: dep,
				}
			}
		}
	}

	for _, rule := range g.Rulemap {
		if err := semvalAlternation(rule.Alternation); err != nil {
			return err
		}
	}
	return nil
}

func semvalAlternation(alt Alternation) error {
	for _, concat := range alt.Concatenations {
		for _, rep := range concat.Repetitions {
			// min <= max
			if rep.Max != inf && rep.Min > rep.Max {
				return &ErrSemanticRepetition{
					Repetition: rep,
				}
			}
			switch elem := rep.Element.(type) {
			// num-val base
			case ElemNumVal:
				for _, val := range elem.Elems {
					if err := checkBounds(val, elem.Base); err != nil {
						return err
					}
				}

			// propagate recursion
			case ElemGroup:
				if err := semvalAlternation(elem.Alternation); err != nil {
					return err
				}

			case ElemOption:
				if err := semvalAlternation(elem.Alternation); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
