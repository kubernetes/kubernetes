package syntax

//go:generate stringer -type=Operation -trimprefix=Op
const (
	OpNone Operation = iota

	// OpConcat is a concatenation of ops.
	// Examples: `xy` `abc\d` ``
	// Args - concatenated ops
	//
	// As a special case, OpConcat with 0 Args is used for "empty"
	// set of operations.
	OpConcat

	// OpDot is a '.' wildcard.
	OpDot

	// OpAlt is x|y alternation of ops.
	// Examples: `a|bc` `x(.*?)|y(.*?)`
	// Args - union-connected regexp branches
	OpAlt

	// OpStar is a shorthand for {0,} repetition.
	// Examples: `x*`
	// Args[0] - repeated expression
	OpStar

	// OpPlus is a shorthand for {1,} repetition.
	// Examples: `x+`
	// Args[0] - repeated expression
	OpPlus

	// OpQuestion is a shorthand for {0,1} repetition.
	// Examples: `x?`
	// Args[0] - repeated expression
	OpQuestion

	// OpNonGreedy makes its operand quantifier non-greedy.
	// Examples: `x??` `x*?` `x+?`
	// Args[0] - quantified expression
	OpNonGreedy

	// OpPossessive makes its operand quantifier possessive.
	// Examples: `x?+` `x*+` `x++`
	// Args[0] - quantified expression
	OpPossessive

	// OpCaret is ^ anchor.
	OpCaret

	// OpDollar is $ anchor.
	OpDollar

	// OpLiteral is a collection of consecutive chars.
	// Examples: `ab` `10x`
	// Args - enclosed characters (OpChar)
	OpLiteral

	// OpChar is a single literal pattern character.
	// Examples: `a` `6` `ф`
	OpChar

	// OpString is an artificial element that is used in other expressions.
	OpString

	// OpQuote is a \Q...\E enclosed literal.
	// Examples: `\Q.?\E` `\Q?q[]=1`
	//
	// Note that closing \E is not mandatory.
	OpQuote

	// OpEscapeChar is a single char escape.
	// Examples: `\d` `\a` `\n`
	OpEscapeChar

	// OpEscapeMeta is an escaped meta char.
	// Examples: `\(` `\[` `\+`
	OpEscapeMeta

	// OpEscapeOctal is an octal char code escape (up to 3 digits).
	// Examples: `\123` `\12`
	OpEscapeOctal

	// OpEscapeHex is a hex char code escape.
	// Examples: `\x7F` `\xF7`
	// FormEscapeHexFull examples: `\x{10FFFF}` `\x{F}`.
	OpEscapeHex

	// OpEscapeUni is a Unicode char class escape.
	// Examples: `\pS` `\pL` `\PL`
	// FormEscapeUniFull examples: `\p{Greek}` `\p{Symbol}` `\p{^L}`
	OpEscapeUni

	// OpCharClass is a char class enclosed in [].
	// Examples: `[abc]` `[a-z0-9\]]`
	// Args - char class elements (can include OpCharRange and OpPosixClass).
	OpCharClass

	// OpNegCharClass is a negated char class enclosed in [].
	// Examples: `[^abc]` `[^a-z0-9\]]`
	// Args - char class elements (can include OpCharRange and OpPosixClass).
	OpNegCharClass

	// OpCharRange is an inclusive char range inside a char class.
	// Examples: `0-9` `A-Z`
	// Args[0] - range lower bound (OpChar or OpEscape).
	// Args[1] - range upper bound (OpChar or OpEscape).
	OpCharRange

	// OpPosixClass is a named ASCII char set inside a char class.
	// Examples: `[:alpha:]` `[:blank:]`
	OpPosixClass

	// OpRepeat is a {min,max} repetition quantifier.
	// Examples: `x{5}` `x{min,max}` `x{min,}`
	// Args[0] - repeated expression
	// Args[1] - repeat count (OpString)
	OpRepeat

	// OpCapture is `(re)` capturing group.
	// Examples: `(abc)` `(x|y)`
	// Args[0] - enclosed expression
	OpCapture

	// OpNamedCapture is `(?P<name>re)` capturing group.
	// Examples: `(?P<foo>abc)` `(?P<name>x|y)`
	// FormNamedCaptureAngle examples: `(?<foo>abc)` `(?<name>x|y)`
	// FormNamedCaptureQuote examples: `(?'foo'abc)` `(?'name'x|y)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	// Args[1] - group name (OpString)
	OpNamedCapture

	// OpGroup is `(?:re)` non-capturing group.
	// Examples: `(?:abc)` `(?:x|y)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpGroup

	// OpGroupWithFlags is `(?flags:re)` non-capturing group.
	// Examples: `(?i:abc)` `(?i:x|y)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	// Args[1] - flags (OpString)
	OpGroupWithFlags

	// OpAtomicGroup is `(?>re)` non-capturing group without backtracking.
	// Examples: `(?>foo)` `(?>)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpAtomicGroup

	// OpPositiveLookahead is `(?=re)` asserts that following text matches re.
	// Examples: `(?=foo)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpPositiveLookahead

	// OpNegativeLookahead is `(?!re)` asserts that following text doesn't match re.
	// Examples: `(?!foo)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpNegativeLookahead

	// OpPositiveLookbehind is `(?<=re)` asserts that preceding text matches re.
	// Examples: `(?<=foo)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpPositiveLookbehind

	// OpNegativeLookbehind is `(?=re)` asserts that preceding text doesn't match re.
	// Examples: `(?<!foo)`
	// Args[0] - enclosed expression (OpConcat with 0 args for empty group)
	OpNegativeLookbehind

	// OpFlagOnlyGroup is `(?flags)` form that affects current group flags.
	// Examples: `(?i)` `(?i-m)` `(?-im)`
	// Args[0] - flags (OpString)
	OpFlagOnlyGroup

	// OpComment is a group-like regexp comment expression.
	// Examples: `(?#text)` `(?#)`
	OpComment

	// OpNone2 is a sentinel value that is never part of the AST.
	// OpNone and OpNone2 can be used to cover all ops in a range.
	OpNone2
)

const (
	FormDefault Form = iota
	FormEscapeHexFull
	FormEscapeUniFull
	FormNamedCaptureAngle
	FormNamedCaptureQuote
)
