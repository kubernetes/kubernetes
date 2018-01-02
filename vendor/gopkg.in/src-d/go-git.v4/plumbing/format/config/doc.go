// Package config implements encoding and decoding of git config files.
//
// 	Configuration File
// 	------------------
//
// 	The Git configuration file contains a number of variables that affect
// 	the Git commands' behavior. The `.git/config` file in each repository
// 	is used to store the configuration for that repository, and
// 	`$HOME/.gitconfig` is used to store a per-user configuration as
// 	fallback values for the `.git/config` file. The file `/etc/gitconfig`
// 	can be used to store a system-wide default configuration.
//
// 	The configuration variables are used by both the Git plumbing
// 	and the porcelains. The variables are divided into sections, wherein
// 	the fully qualified variable name of the variable itself is the last
// 	dot-separated segment and the section name is everything before the last
// 	dot. The variable names are case-insensitive, allow only alphanumeric
// 	characters and `-`, and must start with an alphabetic character.  Some
// 	variables may appear multiple times; we say then that the variable is
// 	multivalued.
//
// 	Syntax
// 	~~~~~~
//
// 	The syntax is fairly flexible and permissive; whitespaces are mostly
// 	ignored.  The '#' and ';' characters begin comments to the end of line,
// 	blank lines are ignored.
//
// 	The file consists of sections and variables.  A section begins with
// 	the name of the section in square brackets and continues until the next
// 	section begins.  Section names are case-insensitive.  Only alphanumeric
// 	characters, `-` and `.` are allowed in section names.  Each variable
// 	must belong to some section, which means that there must be a section
// 	header before the first setting of a variable.
//
// 	Sections can be further divided into subsections.  To begin a subsection
// 	put its name in double quotes, separated by space from the section name,
// 	in the section header, like in the example below:
//
// 	--------
// 		[section "subsection"]
//
// 	--------
//
// 	Subsection names are case sensitive and can contain any characters except
// 	newline (doublequote `"` and backslash can be included by escaping them
// 	as `\"` and `\\`, respectively).  Section headers cannot span multiple
// 	lines.  Variables may belong directly to a section or to a given subsection.
// 	You can have `[section]` if you have `[section "subsection"]`, but you
// 	don't need to.
//
// 	There is also a deprecated `[section.subsection]` syntax. With this
// 	syntax, the subsection name is converted to lower-case and is also
// 	compared case sensitively. These subsection names follow the same
// 	restrictions as section names.
//
// 	All the other lines (and the remainder of the line after the section
// 	header) are recognized as setting variables, in the form
// 	'name = value' (or just 'name', which is a short-hand to say that
// 	the variable is the boolean "true").
// 	The variable names are case-insensitive, allow only alphanumeric characters
// 	and `-`, and must start with an alphabetic character.
//
// 	A line that defines a value can be continued to the next line by
// 	ending it with a `\`; the backquote and the end-of-line are
// 	stripped.  Leading whitespaces after 'name =', the remainder of the
// 	line after the first comment character '#' or ';', and trailing
// 	whitespaces of the line are discarded unless they are enclosed in
// 	double quotes.  Internal whitespaces within the value are retained
// 	verbatim.
//
// 	Inside double quotes, double quote `"` and backslash `\` characters
// 	must be escaped: use `\"` for `"` and `\\` for `\`.
//
// 	The following escape sequences (beside `\"` and `\\`) are recognized:
// 	`\n` for newline character (NL), `\t` for horizontal tabulation (HT, TAB)
// 	and `\b` for backspace (BS).  Other char escape sequences (including octal
// 	escape sequences) are invalid.
//
// 	Includes
// 	~~~~~~~~
//
// 	You can include one config file from another by setting the special
// 	`include.path` variable to the name of the file to be included. The
// 	variable takes a pathname as its value, and is subject to tilde
// 	expansion.
//
// 	The included file is expanded immediately, as if its contents had been
// 	found at the location of the include directive. If the value of the
// 	`include.path` variable is a relative path, the path is considered to be
// 	relative to the configuration file in which the include directive was
// 	found.  See below for examples.
//
//
// 	Example
// 	~~~~~~~
//
// 		# Core variables
// 		[core]
// 			; Don't trust file modes
// 			filemode = false
//
// 		# Our diff algorithm
// 		[diff]
// 			external = /usr/local/bin/diff-wrapper
// 			renames = true
//
// 		[branch "devel"]
// 			remote = origin
// 			merge = refs/heads/devel
//
// 		# Proxy settings
// 		[core]
// 			gitProxy="ssh" for "kernel.org"
// 			gitProxy=default-proxy ; for the rest
//
// 		[include]
// 			path = /path/to/foo.inc ; include by absolute path
// 			path = foo ; expand "foo" relative to the current file
// 			path = ~/foo ; expand "foo" in your `$HOME` directory
//
package config
