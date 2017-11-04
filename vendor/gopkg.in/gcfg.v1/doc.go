// Package gcfg reads "INI-style" text-based configuration files with
// "name=value" pairs grouped into sections (gcfg files).
//
// This package is still a work in progress; see the sections below for planned
// changes.
//
// Syntax
//
// The syntax is based on that used by git config:
// http://git-scm.com/docs/git-config#_syntax .
// There are some (planned) differences compared to the git config format:
//  - improve data portability:
//    - must be encoded in UTF-8 (for now) and must not contain the 0 byte
//    - include and "path" type is not supported
//      (path type may be implementable as a user-defined type)
//  - internationalization
//    - section and variable names can contain unicode letters, unicode digits
//      (as defined in http://golang.org/ref/spec#Characters ) and hyphens
//      (U+002D), starting with a unicode letter
//  - disallow potentially ambiguous or misleading definitions:
//    - `[sec.sub]` format is not allowed (deprecated in gitconfig)
//    - `[sec ""]` is not allowed
//      - use `[sec]` for section name "sec" and empty subsection name
//    - (planned) within a single file, definitions must be contiguous for each:
//      - section: '[secA]' -> '[secB]' -> '[secA]' is an error
//      - subsection: '[sec "A"]' -> '[sec "B"]' -> '[sec "A"]' is an error
//      - multivalued variable: 'multi=a' -> 'other=x' -> 'multi=b' is an error
//
// Data structure
//
// The functions in this package read values into a user-defined struct.
// Each section corresponds to a struct field in the config struct, and each
// variable in a section corresponds to a data field in the section struct.
// The mapping of each section or variable name to fields is done either based
// on the "gcfg" struct tag or by matching the name of the section or variable,
// ignoring case. In the latter case, hyphens '-' in section and variable names
// correspond to underscores '_' in field names.
// Fields must be exported; to use a section or variable name starting with a
// letter that is neither upper- or lower-case, prefix the field name with 'X'.
// (See https://code.google.com/p/go/issues/detail?id=5763#c4 .)
//
// For sections with subsections, the corresponding field in config must be a
// map, rather than a struct, with string keys and pointer-to-struct values.
// Values for subsection variables are stored in the map with the subsection
// name used as the map key.
// (Note that unlike section and variable names, subsection names are case
// sensitive.)
// When using a map, and there is a section with the same section name but
// without a subsection name, its values are stored with the empty string used
// as the key.
// It is possible to provide default values for subsections in the section
// "default-<sectionname>" (or by setting values in the corresponding struct
// field "Default_<sectionname>").
//
// The functions in this package panic if config is not a pointer to a struct,
// or when a field is not of a suitable type (either a struct or a map with
// string keys and pointer-to-struct values).
//
// Parsing of values
//
// The section structs in the config struct may contain single-valued or
// multi-valued variables. Variables of unnamed slice type (that is, a type
// starting with `[]`) are treated as multi-value; all others (including named
// slice types) are treated as single-valued variables.
//
// Single-valued variables are handled based on the type as follows.
// Unnamed pointer types (that is, types starting with `*`) are dereferenced,
// and if necessary, a new instance is allocated.
//
// For types implementing the encoding.TextUnmarshaler interface, the
// UnmarshalText method is used to set the value. Implementing this method is
// the recommended way for parsing user-defined types.
//
// For fields of string kind, the value string is assigned to the field, after
// unquoting and unescaping as needed.
// For fields of bool kind, the field is set to true if the value is "true",
// "yes", "on" or "1", and set to false if the value is "false", "no", "off" or
// "0", ignoring case. In addition, single-valued bool fields can be specified
// with a "blank" value (variable name without equals sign and value); in such
// case the value is set to true.
//
// Predefined integer types [u]int(|8|16|32|64) and big.Int are parsed as
// decimal or hexadecimal (if having '0x' prefix). (This is to prevent
// unintuitively handling zero-padded numbers as octal.) Other types having
// [u]int* as the underlying type, such as os.FileMode and uintptr allow
// decimal, hexadecimal, or octal values.
// Parsing mode for integer types can be overridden using the struct tag option
// ",int=mode" where mode is a combination of the 'd', 'h', and 'o' characters
// (each standing for decimal, hexadecimal, and octal, respectively.)
//
// All other types are parsed using fmt.Sscanf with the "%v" verb.
//
// For multi-valued variables, each individual value is parsed as above and
// appended to the slice. If the first value is specified as a "blank" value
// (variable name without equals sign and value), a new slice is allocated;
// that is any values previously set in the slice will be ignored.
//
// The types subpackage for provides helpers for parsing "enum-like" and integer
// types.
//
// Error handling
//
// There are 3 types of errors:
//
//  - programmer errors / panics:
//    - invalid configuration structure
//  - data errors:
//    - fatal errors:
//      - invalid configuration syntax
//    - warnings:
//      - data that doesn't belong to any part of the config structure
//
// Programmer errors trigger panics. These are should be fixed by the programmer
// before releasing code that uses gcfg.
//
// Data errors cause gcfg to return a non-nil error value. This includes the
// case when there are extra unknown key-value definitions in the configuration
// data (extra data).
// However, in some occasions it is desirable to be able to proceed in
// situations when the only data error is that of extra data.
// These errors are handled at a different (warning) priority and can be
// filtered out programmatically. To ignore extra data warnings, wrap the
// gcfg.Read*Into invocation into a call to gcfg.FatalOnly.
//
// TODO
//
// The following is a list of changes under consideration:
//  - documentation
//    - self-contained syntax documentation
//    - more practical examples
//    - move TODOs to issue tracker (eventually)
//  - syntax
//    - reconsider valid escape sequences
//      (gitconfig doesn't support \r in value, \t in subsection name, etc.)
//  - reading / parsing gcfg files
//    - define internal representation structure
//    - support multiple inputs (readers, strings, files)
//    - support declaring encoding (?)
//    - support varying fields sets for subsections (?)
//  - writing gcfg files
//  - error handling
//    - make error context accessible programmatically?
//    - limit input size?
//
package gcfg
