# HIL

[![GoDoc](https://godoc.org/github.com/hashicorp/hil?status.png)](https://godoc.org/github.com/hashicorp/hil) [![Build Status](https://travis-ci.org/hashicorp/hil.svg?branch=master)](https://travis-ci.org/hashicorp/hil)

HIL (HashiCorp Interpolation Language) is a lightweight embedded language used
primarily for configuration interpolation. The goal of HIL is to make a simple
language for interpolations in the various configurations of HashiCorp tools.

HIL is built to interpolate any string, but is in use by HashiCorp primarily
with [HCL](https://github.com/hashicorp/hcl). HCL is _not required_ in any
way for use with HIL.

HIL isn't meant to be a general purpose language. It was built for basic
configuration interpolations. Therefore, you can't currently write functions,
have conditionals, set intermediary variables, etc. within HIL itself. It is
possible some of these may be added later but the right use case must exist.

## Why?

Many of our tools have support for something similar to templates, but
within the configuration itself. The most prominent requirement was in
[Terraform](https://github.com/hashicorp/terraform) where we wanted the
configuration to be able to reference values from elsewhere in the
configuration. Example:

    foo = "hi ${var.world}"

We originally used a full templating language for this, but found it
was too heavy weight. Additionally, many full languages required bindings
to C (and thus the usage of cgo) which we try to avoid to make cross-compilation
easier. We then moved to very basic regular expression based
string replacement, but found the need for basic arithmetic and function
calls resulting in overly complex regular expressions.

Ultimately, we wrote our own mini-language within Terraform itself. As
we built other projects such as [Nomad](https://nomadproject.io) and
[Otto](https://ottoproject.io), the need for basic interpolations arose
again.

Thus HIL was born. It is extracted from Terraform, cleaned up, and
better tested for general purpose use.

## Syntax

For a complete grammar, please see the parser itself. A high-level overview
of the syntax and grammer is listed here.

Code begins within `${` and `}`. Outside of this, text is treated
literally. For example, `foo` is a valid HIL program that is just the
string "foo", but `foo ${bar}` is an HIL program that is the string "foo "
concatened with the value of `bar`. For the remainder of the syntax
docs, we'll assume you're within `${}`.

  * Identifiers are any text in the format of `[a-zA-Z0-9-.]`. Example
    identifiers: `foo`, `var.foo`, `foo-bar`.

  * Strings are double quoted and can contain any UTF-8 characters.
    Example: `"Hello, World"`

  * Numbers are assumed to be base 10. If you prefix a number with 0x,
    it is treated as a hexadecimal. If it is prefixed with 0, it is
    treated as an octal. Numbers can be in scientific notation: "1e10".

  * Unary `-` can be used for negative numbers. Example: `-10` or `-0.2`

  * Boolean values: `true`, `false`
  
  * The following arithmetic operations are allowed: +, -, *, /, %. 

  * Function calls are in the form of `name(arg1, arg2, ...)`. Example:
    `add(1, 5)`. Arguments can be any valid HIL expression, example:
    `add(1, var.foo)` or even nested function calls:
    `add(1, get("some value"))`. 

  * Within strings, further interpolations can be opened with `${}`.
    Example: `"Hello ${nested}"`. A full example including the 
    original `${}` (remember this list assumes were inside of one
    already) could be: `foo ${func("hello ${var.foo}")}`. 

## Language Changes

We've used this mini-language in Terraform for years. For backwards compatibility
reasons, we're unlikely to make an incompatible change to the language but
we're not currently making that promise, either.

The internal API of this project may very well change as we evolve it
to work with more of our projects. We recommend using some sort of dependency
management solution with this package.

## Future Changes

The following changes are already planned to be made at some point:

  * Richer types: lists, maps, etc.

  * Convert to a more standard Go parser structure similar to HCL. This
    will improve our error messaging as well as allow us to have automatic
    formatting.

  * Allow interpolations to result in more types than just a string. While
    within the interpolation basic types are honored, the result is always
    a string.
