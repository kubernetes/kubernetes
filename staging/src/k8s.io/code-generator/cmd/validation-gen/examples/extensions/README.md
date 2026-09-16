# Adding formats with `--validation-extensions-file`

A project that needs a string rule the built-in tags do not cover can write it
as `+k8s:customValidation` and a function per use, or declare it once as a
**format** and use `+k8s:format=<name>` wherever it applies. A format composes
with `eachVal`, `eachKey`, `ifMode` and `ifEnabled` and carries an origin, so
its errors are attributable.

Formats used to be a closed list. `--validation-extensions-file` opens it, using
the **stock** generator rather than one built for the project:

```
validation-gen --validation-extensions-file=path/to/extensions.yaml ...
```

The flag may be repeated. Each file is one YAML document, formats from every
file are concatenated, and a name may only be declared once across all of them.

[extensions.yaml](extensions.yaml) is this example's file,
[output_tests/generate.go](output_tests/generate.go) is the invocation that
uses it, and the fields are documented on `validators.FormatExtension`. Compare
with [../custom-prefix](../custom-prefix), which is what a project has to build
when it needs something this file cannot express.

Three things that file does not show: a pattern is **not implicitly anchored**,
so write `^` and `$` if that is what is meant; the generator compiles each
pattern into one package-level variable, however many fields use it; and the
`k8s-` prefix is reserved, so a project cannot take a name a later release
might want.

## Writing patterns

Single-quote them. YAML takes a single-quoted scalar literally, which is what a
regular expression wants:

```yaml
pattern: '^[a-z]+://[^\s]+$'   # literal
pattern: "^[a-z]+://[^\s]+$"   # error: found unknown escape character
pattern: "^[a-z]+://[^\\s]+$"  # works, but every backslash has to be doubled
pattern: ^[a-z]+://[^\s]+$     # works here, but `[a-z]+` alone is a YAML list
```

The same applies to KYAML, which parses fine here because it is a subset of
YAML, but which always double-quotes and so needs the doubled backslashes.

## Errors

Every file is checked once, before generation, so a mistake names its source
rather than surfacing at the first use of the tag:

```
Failed loading validation extensions: extensions file "extensions.yaml":
  formats[0] ("example-uri"): pattern "[unterminated" does not compile: ...
```

A project may not redefine a built-in format, so a tag means the same thing in
every project that recognizes it. `validation-gen --docs
--validation-extensions-file=extensions.yaml` lists the project's formats
alongside the built-in ones.

## Not yet

A pattern is the only backing a format has today. A shared Go function, a CEL
expression and length bounds are all keys beside `pattern` rather than a new
shape for the file, and `formats` is one key at the top level rather than the
root, so named validation rules can be added later as a sibling section.
