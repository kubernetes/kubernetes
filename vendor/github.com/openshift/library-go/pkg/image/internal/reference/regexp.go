package reference

import "regexp"

var (
	// alphaNumericRegexp defines the alpha numeric atom, typically a
	// component of names. This only allows lower case characters and digits.
	alphaNumericRegexp = match(`[a-z0-9]+`)

	// separatorRegexp defines the separators allowed to be embedded in name
	// components. This allow one period, one or two underscore and multiple
	// dashes.
	separatorRegexp = match(`(?:[._]|__|[-]*)`)

	// nameComponentRegexp restricts registry path component names to start
	// with at least one letter or number, with following parts able to be
	// separated by one period, one or two underscore and multiple dashes.
	nameComponentRegexp = expression(
		alphaNumericRegexp,
		optional(repeated(separatorRegexp, alphaNumericRegexp)))

	// hostnameComponentRegexp restricts the registry hostname component of a
	// repository name to start with a component as defined by hostnameRegexp
	// and followed by an optional port.
	hostnameComponentRegexp = match(`(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])`)

	// hostnameRegexp defines the structure of potential hostname components
	// that may be part of image names. This is purposely a subset of what is
	// allowed by DNS to ensure backwards compatibility with Docker image
	// names.
	hostnameRegexp = expression(
		hostnameComponentRegexp,
		optional(repeated(literal(`.`), hostnameComponentRegexp)),
		optional(literal(`:`), match(`[0-9]+`)))

	// TagRegexp matches valid tag names. From docker/docker:graph/tags.go.
	TagRegexp = match(`[\w][\w.-]{0,127}`)

	// anchoredTagRegexp matches valid tag names, anchored at the start and
	// end of the matched string.
	anchoredTagRegexp = anchored(TagRegexp)

	// DigestRegexp matches valid digests.
	DigestRegexp = match(`[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*[:][[:xdigit:]]{32,}`)

	// anchoredDigestRegexp matches valid digests, anchored at the start and
	// end of the matched string.
	anchoredDigestRegexp = anchored(DigestRegexp)

	// NameRegexp is the format for the name component of references. The
	// regexp has capturing groups for the hostname and name part omitting
	// the separating forward slash from either.
	NameRegexp = expression(
		optional(hostnameRegexp, literal(`/`)),
		nameComponentRegexp,
		optional(repeated(literal(`/`), nameComponentRegexp)))

	// anchoredNameRegexp is used to parse a name value, capturing the
	// hostname and trailing components.
	anchoredNameRegexp = anchored(
		optional(capture(hostnameRegexp), literal(`/`)),
		capture(nameComponentRegexp,
			optional(repeated(literal(`/`), nameComponentRegexp))))

	// ReferenceRegexp is the full supported format of a reference. The regexp
	// is anchored and has capturing groups for name, tag, and digest
	// components.
	ReferenceRegexp = anchored(capture(NameRegexp),
		optional(literal(":"), capture(TagRegexp)),
		optional(literal("@"), capture(DigestRegexp)))
)

// match compiles the string to a regular expression.
var match = regexp.MustCompile

// literal compiles s into a literal regular expression, escaping any regexp
// reserved characters.
func literal(s string) *regexp.Regexp {
	re := match(regexp.QuoteMeta(s))

	if _, complete := re.LiteralPrefix(); !complete {
		panic("must be a literal")
	}

	return re
}

// expression defines a full expression, where each regular expression must
// follow the previous.
func expression(res ...*regexp.Regexp) *regexp.Regexp {
	var s string
	for _, re := range res {
		s += re.String()
	}

	return match(s)
}

// optional wraps the expression in a non-capturing group and makes the
// production optional.
func optional(res ...*regexp.Regexp) *regexp.Regexp {
	return match(group(expression(res...)).String() + `?`)
}

// repeated wraps the regexp in a non-capturing group to get one or more
// matches.
func repeated(res ...*regexp.Regexp) *regexp.Regexp {
	return match(group(expression(res...)).String() + `+`)
}

// group wraps the regexp in a non-capturing group.
func group(res ...*regexp.Regexp) *regexp.Regexp {
	return match(`(?:` + expression(res...).String() + `)`)
}

// capture wraps the expression in a capturing group.
func capture(res ...*regexp.Regexp) *regexp.Regexp {
	return match(`(` + expression(res...).String() + `)`)
}

// anchored anchors the regular expression by adding start and end delimiters.
func anchored(res ...*regexp.Regexp) *regexp.Regexp {
	return match(`^` + expression(res...).String() + `$`)
}
