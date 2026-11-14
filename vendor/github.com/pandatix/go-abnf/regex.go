package goabnf

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// Regex builds a regular expression that recognises the given rulename.
//
// The rule must not contain a cycle in its dependency graph: a regular
// expression (RE2, no subroutines) cannot express recursion.
//
// Because RE2 has no subroutine/backreference, a rule referenced N times is
// inlined N times, so the output can grow exponentially with reference-nesting
// depth for diamond-shaped grammars. Pass WithMaxRegexLen to fail fast with
// *ErrRegexTooLarge instead of exhausting memory. The output is simplified
// (character classes merged, redundant groups and {1} quantifiers dropped) but
// is not guaranteed minimal.
func (g *Grammar) Regex(rulename string, opts ...RegexOption) (string, error) {
	o := &regexOptions{}
	for _, op := range opts {
		op.applyRegex(o)
	}

	isCyclic, err := g.RuleContainsCycle(rulename)
	if err != nil {
		return "", err
	}
	if isCyclic {
		return "", &ErrCyclicRule{Rulename: rulename}
	}
	rule := GetRule(rulename, g.Rulemap)
	if rule == nil {
		return "", &ErrRuleNotFound{Rulename: rulename}
	}

	b := &reBuilder{g: g, memo: map[string]reNode{}}
	node := b.alt(rule.Alternation)
	if b.err != nil {
		return "", b.err
	}
	node = simplify(node, map[reNode]reNode{})

	r := &reRenderer{max: o.maxLen}
	var sb strings.Builder
	r.render(&sb, node, precAlt)
	if r.err != nil {
		return "", r.err
	}
	return sb.String(), nil
}

// ---- options & error ----

type regexOptions struct{ maxLen int }

// RegexOption configures Regex.
type RegexOption interface{ applyRegex(*regexOptions) }

type regexOptionFunc func(*regexOptions)

func (f regexOptionFunc) applyRegex(o *regexOptions) { f(o) }

// WithMaxRegexLen bounds the produced regex length in bytes. When the budget is
// exceeded, Regex returns *ErrRegexTooLarge. Zero (the default) is unbounded.
func WithMaxRegexLen(n int) RegexOption {
	return regexOptionFunc(func(o *regexOptions) { o.maxLen = n })
}

// ErrRegexTooLarge is returned by Regex when the output would exceed the budget
// set with WithMaxRegexLen.
type ErrRegexTooLarge struct{ Max int }

var _ error = (*ErrRegexTooLarge)(nil)

func (e *ErrRegexTooLarge) Error() string {
	return fmt.Sprintf("regex exceeds the configured %d-byte budget", e.Max)
}

// ---- intermediate representation ----

const (
	precAlt    = 1 // alternation binds loosest
	precConcat = 2
	precRepeat = 3
	precAtom   = 4 // a class or single char binds tightest
)

type reNode interface{ prec() int }

type reRange struct{ lo, hi rune }

type reClass struct{ ranges []reRange } // a single [c,c] renders as a bare char
type reConcat struct{ parts []reNode }
type reAlt struct{ opts []reNode }
type reRepeat struct {
	sub      reNode
	min, max int
}
type reEmpty struct{}

// reNever matches nothing. It represents a num-val a RE2/Unicode pattern cannot
// express (a value entirely above U+10FFFF), keeping Regex consistent with the
// recognizer, which also matches nothing in that case.
type reNever struct{}

func (*reClass) prec() int  { return precAtom }
func (*reConcat) prec() int { return precConcat }
func (*reAlt) prec() int    { return precAlt }
func (*reRepeat) prec() int { return precRepeat }
func (*reEmpty) prec() int  { return precAtom }
func (*reNever) prec() int  { return precAtom }

var reEmptyV reNode = &reEmpty{}
var reNeverV reNode = &reNever{}

func isNever(n reNode) bool { _, ok := n.(*reNever); return ok }

// ---- build (memoised per rule => linear over the grammar DAG) ----

type reBuilder struct {
	g    *Grammar
	memo map[string]reNode
	err  error
}

func (b *reBuilder) rule(name string) reNode {
	key := strings.ToLower(name)
	if n, ok := b.memo[key]; ok {
		return n
	}
	r := GetRule(name, b.g.Rulemap)
	if r == nil {
		return reEmptyV
	}
	n := b.alt(r.Alternation)
	b.memo[key] = n
	return n
}

func (b *reBuilder) alt(a Alternation) reNode {
	opts := make([]reNode, 0, len(a.Concatenations))
	for _, c := range a.Concatenations {
		opts = append(opts, b.concat(c))
	}
	return &reAlt{opts: opts}
}

func (b *reBuilder) concat(c Concatenation) reNode {
	parts := make([]reNode, 0, len(c.Repetitions))
	for _, rep := range c.Repetitions {
		parts = append(parts, &reRepeat{sub: b.elem(rep.Element), min: rep.Min, max: rep.Max})
	}
	return &reConcat{parts: parts}
}

func (b *reBuilder) elem(e ElemItf) reNode {
	switch v := e.(type) {
	case ElemRulename:
		return b.rule(v.Name)
	case ElemGroup:
		return b.alt(v.Alternation)
	case ElemOption:
		return &reRepeat{sub: b.alt(v.Alternation), min: 0, max: 1}
	case ElemNumVal:
		switch v.Status {
		case StatRange:
			lo := numvalToRune(v.Elems[0], v.Base)
			hi := min(
				// RE2 patterns are Unicode (max U+10FFFF). A range straddling that
				// ceiling is clamped to its representable subset; one wholly above it
				// matches nothing -- mirroring the recognizer.
				numvalToRune(v.Elems[1], v.Base), utf8.MaxRune)
			if lo > utf8.MaxRune || lo > hi {
				return reNeverV
			}
			return &reClass{ranges: []reRange{{lo, hi}}}
		default: // StatSeries is a CONCATENATION of values (a fixed sequence).
			parts := make([]reNode, 0, len(v.Elems))
			for _, s := range v.Elems {
				c := numvalToRune(s, v.Base)
				// A series element must be a real character; if it is not, the
				// whole sequence can never match any (Unicode) input.
				if !utf8.ValidRune(c) {
					return reNeverV
				}
				parts = append(parts, &reClass{ranges: []reRange{{c, c}}})
			}
			return &reConcat{parts: parts}
		}
	case ElemCharVal:
		// A char-val is case-insensitive unless Sensitive (RFC 7405); the
		// recognizer folds ASCII A-Z only (see runeMin/sensequal), so we match
		// that exactly here.
		parts := make([]reNode, 0, len(v.Values))
		for _, c := range v.Values {
			parts = append(parts, charClass(c, v.Sensitive))
		}
		return &reConcat{parts: parts}
	case ElemProseVal:
		if b.err == nil {
			b.err = ErrHandlingProseVal
		}
		return reEmptyV
	}
	return reEmptyV
}

func charClass(c rune, sensitive bool) reNode {
	if !sensitive {
		switch {
		case c >= 'A' && c <= 'Z':
			lo := c - 'A' + 'a'
			return &reClass{ranges: []reRange{{c, c}, {lo, lo}}}
		case c >= 'a' && c <= 'z':
			up := c - 'a' + 'A'
			return &reClass{ranges: []reRange{{up, up}, {c, c}}}
		}
	}
	return &reClass{ranges: []reRange{{c, c}}}
}

// ---- simplification (pointer-memoised => linear over the DAG) ----

func simplify(n reNode, memo map[reNode]reNode) reNode {
	if n == nil {
		return reEmptyV
	}
	if s, ok := memo[n]; ok {
		return s
	}
	var res reNode
	switch v := n.(type) {
	case *reClass:
		res = &reClass{ranges: normalizeRanges(v.ranges)}
	case *reEmpty:
		res = reEmptyV
	case *reNever:
		res = reNeverV
	case *reConcat:
		parts := make([]reNode, 0, len(v.parts))
		never := false
		for _, p := range v.parts {
			sp := simplify(p, memo)
			switch x := sp.(type) {
			case *reEmpty: // drop: empty in a concatenation contributes nothing
			case *reNever: // one un-matchable part makes the whole sequence un-matchable
				never = true
			case *reConcat:
				parts = append(parts, x.parts...) // flatten
			default:
				parts = append(parts, sp)
			}
		}
		switch {
		case never:
			res = reNeverV
		case len(parts) == 0:
			res = reEmptyV
		case len(parts) == 1:
			res = parts[0]
		default:
			res = &reConcat{parts: parts}
		}
	case *reAlt:
		opts := make([]reNode, 0, len(v.opts))
		for _, o := range v.opts {
			so := simplify(o, memo)
			if isNever(so) { // a never-matching branch is dead; drop it
				continue
			}
			if x, ok := so.(*reAlt); ok {
				opts = append(opts, x.opts...) // flatten
			} else {
				opts = append(opts, so)
			}
		}
		// Fold every class option into a single merged class: turns an
		// alternation of single characters / ranges into one [...].
		var ranges []reRange
		classes := 0
		others := make([]reNode, 0, len(opts))
		for _, o := range opts {
			if c, ok := o.(*reClass); ok {
				ranges = append(ranges, c.ranges...)
				classes++
			} else {
				others = append(others, o)
			}
		}
		merged := make([]reNode, 0, len(others)+1)
		if classes > 0 {
			merged = append(merged, &reClass{ranges: normalizeRanges(ranges)})
		}
		merged = append(merged, others...)
		switch len(merged) {
		case 0:
			// Every branch was un-matchable: the alternation matches nothing.
			res = reNeverV
		case 1:
			res = merged[0]
		default:
			res = &reAlt{opts: merged}
		}
	case *reRepeat:
		sub := simplify(v.sub, memo)
		switch {
		case isNever(sub):
			// Zero occurrences are allowed iff min == 0 (then it matches empty);
			// otherwise the repetition is itself un-matchable.
			if v.min == 0 {
				res = reEmptyV
			} else {
				res = reNeverV
			}
		case isEmpty(sub):
			res = reEmptyV
		case v.min == 1 && v.max == 1:
			res = sub
		default:
			res = &reRepeat{sub: sub, min: v.min, max: v.max}
		}
	default:
		res = n
	}
	memo[n] = res
	return res
}

func isEmpty(n reNode) bool { _, ok := n.(*reEmpty); return ok }

// normalizeRanges sorts and coalesces overlapping/adjacent ranges.
func normalizeRanges(rs []reRange) []reRange {
	if len(rs) <= 1 {
		return rs
	}
	cp := append([]reRange(nil), rs...)
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].lo < cp[j].lo || (cp[i].lo == cp[j].lo && cp[i].hi < cp[j].hi)
	})
	out := []reRange{cp[0]}
	for _, r := range cp[1:] {
		last := &out[len(out)-1]
		if r.lo <= last.hi+1 { // overlapping or adjacent
			if r.hi > last.hi {
				last.hi = r.hi
			}
		} else {
			out = append(out, r)
		}
	}
	return out
}

// ---- render (minimal grouping, length-budgeted) ----

type reRenderer struct {
	max int
	n   int
	err error
}

func (rr *reRenderer) write(sb *strings.Builder, s string) {
	if rr.err != nil {
		return
	}
	rr.n += len(s)
	if rr.max > 0 && rr.n > rr.max {
		rr.err = &ErrRegexTooLarge{Max: rr.max}
		return
	}
	sb.WriteString(s)
}

// render emits node; it self-groups with (?:...) when its precedence is looser
// than the surrounding context requires.
func (rr *reRenderer) render(sb *strings.Builder, n reNode, ctx int) {
	if rr.err != nil {
		return
	}
	grp := n.prec() < ctx
	if grp {
		rr.write(sb, "(?:")
	}
	switch v := n.(type) {
	case *reEmpty:
		// (?: ) already opened if grouped; otherwise emit an explicit empty.
		if !grp {
			rr.write(sb, "(?:)")
		}
	case *reNever:
		// Negated full-Unicode class: matches no character, hence nothing.
		rr.write(sb, "[^\\x00-\\x{10ffff}]")
	case *reClass:
		rr.write(sb, renderClass(v))
	case *reConcat:
		for _, p := range v.parts {
			rr.render(sb, p, precConcat)
		}
	case *reAlt:
		for i, o := range v.opts {
			if i > 0 {
				rr.write(sb, "|")
			}
			rr.render(sb, o, precAlt)
		}
	case *reRepeat:
		rr.render(sb, v.sub, precAtom) // operand of a quantifier must be an atom
		rr.write(sb, quantifier(v.min, v.max))
	}
	if grp {
		rr.write(sb, ")")
	}
}

func renderClass(c *reClass) string {
	if len(c.ranges) == 1 && c.ranges[0].lo == c.ranges[0].hi {
		return regescape(c.ranges[0].lo) // bare single character
	}
	var b strings.Builder
	b.WriteByte('[')
	for _, r := range c.ranges {
		if r.lo == r.hi {
			b.WriteString(classEscape(r.lo))
		} else {
			b.WriteString(classEscape(r.lo))
			b.WriteByte('-')
			b.WriteString(classEscape(r.hi))
		}
	}
	b.WriteByte(']')
	return b.String()
}

func quantifier(min, max int) string {
	switch {
	case min == 0 && max == 1:
		return "?"
	case min == 0 && max == inf:
		return "*"
	case min == 1 && max == inf:
		return "+"
	case max == inf:
		return fmt.Sprintf("{%d,}", min)
	case min == max:
		return fmt.Sprintf("{%d}", min)
	case min == 0:
		return fmt.Sprintf("{0,%d}", max)
	default:
		return fmt.Sprintf("{%d,%d}", min, max)
	}
}

// regescape quotes a single rune for use outside a character class.
func regescape(r rune) string { return regexp.QuoteMeta(string(r)) }

// classEscape renders a rune as a character-class member. Alphanumerics are
// never class metacharacters; everything else is emitted as \x{...} to avoid
// the class-specific escaping pitfalls of ], ^, - and \.
func classEscape(r rune) string {
	if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
		return string(r)
	}
	return fmt.Sprintf("\\x{%x}", r)
}
