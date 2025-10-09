package spiffeid

import "errors"

var (
	errBadTrustDomainChar = errors.New("trust domain characters are limited to lowercase letters, numbers, dots, dashes, and underscores")
	errBadPathSegmentChar = errors.New("path segment characters are limited to letters, numbers, dots, dashes, and underscores")
	errDotSegment         = errors.New("path cannot contain dot segments")
	errNoLeadingSlash     = errors.New("path must have a leading slash")
	errEmpty              = errors.New("cannot be empty")
	errEmptySegment       = errors.New("path cannot contain empty segments")
	errMissingTrustDomain = errors.New("trust domain is missing")
	errTrailingSlash      = errors.New("path cannot have a trailing slash")
	errWrongScheme        = errors.New("scheme is missing or invalid")
)
