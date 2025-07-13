# Content validation

This package holds functions which validate the "contents" of various types
(often strings, but not exclusively) against defined rules.  It is intended for
use by Kubernetes API validation code, but is loosely-coupled so it can, in
theory, be more useful.

Most of the public functions here return a `[]string` where each item is a
distinct validation failure message and a zero-length return value (not just
nil) indicates success.  Many of these functions will return just one error,
but they should still use a slice return type.

Good validation failure messages follow the Kubernetes API conventions, for
example using "must" instead of "should".

This package should have almost no external dependecies (except stdlib).
