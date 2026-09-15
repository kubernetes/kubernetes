# Adding formats with `--profile`

A project that needs a string rule the built-in tags do not cover can write it
as `+k8s:customValidation` and a function per use, or declare it once as a
**format** and use `+k8s:format=<name>` wherever it applies. A format composes
with `eachVal`, `eachKey` and `ifEnabled` and carries an origin, so its errors
are attributable.

Formats used to be a closed list. `--profile` opens it, using the **stock**
generator rather than one built for the project:

```
validation-gen --profile=path/to/profile.yaml ...
```

[profile.yaml](profile.yaml) is this example's profile,
[output_tests/generate.go](output_tests/generate.go) is the invocation that
uses it, and the fields are documented on `validators.ProfileFormat`. Compare
with [../custom-prefix](../custom-prefix), which is what a project has to build
when it needs something a profile cannot express.

Two things that file does not show: a pattern is **not implicitly anchored**,
so write `^` and `$` if that is what is meant; and the generator compiles each
pattern into one package-level variable, however many fields use it.

## Errors

A profile is checked once, before generation, so a mistake in it names the
format rather than surfacing at the first use of the tag:

```
Failed loading profile: profile "profile.yaml": formats[0] ("example-uri"):
  regex "[unterminated" does not compile: error parsing regexp: ...
```

A profile may not redefine a built-in format, so a tag means the same thing in
every project that recognizes it.
`validation-gen --docs --profile=profile.yaml` lists the project's formats
alongside the built-in ones.

## Not yet

A regex is the only backing a format has today. A shared Go function, a CEL
expression and length bounds are all keys beside `regex` rather than a new
shape for the file, and `formats` is one key at the top level rather than the
root, so named validations can be added later as a sibling section.
