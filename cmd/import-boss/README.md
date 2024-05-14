## Purpose

`import-boss` enforces optional import restrictions between packages.  This is
useful to manage the dependency graph within a large repository, such as
[kubernetes](https://github.com/kubernetes/kubernetes).

## How does it work?

When a package is verified, `import-boss` looks for a file called
`.import-restrictions` in the same directory and all parent directories, up to
the module root (defined by the presence of a go.mod file).  These files
contain rules which are evaluated against each dependency of the package in
question.

Evaluation starts with the rules file closest to the package.  If that file
makes a determination to allow or forbid the import, evaluation is done.  If
the import does not match any rule, the next-closest rules file is consulted,
and so forth.  If the rules files are exhausted and no determination has been
made, the import will be flagged as an error.

### What are rules files?

A rules file is a JSON or YAML document with two top-level keys, both optional:
* `Rules`
* `InverseRules`

### What are Rules?

A `rule` defines a policy to be enforced on packages which are depended on by
the package in question.  It consists of three parts:
  - A `SelectorRegexp`, to select the import paths that the rule applies to.
  - A list of `AllowedPrefixes`
  - A list of `ForbiddenPrefixes`

An import is allowed if it matches at least one allowed prefix and does not
match any forbidden prefixes.

Rules also have a boolean `Transitive` option. When this option is true, the
rule is applied to transitive imports.

Example:

```json
{
  "Rules": [
    {
      "SelectorRegexp": "example[.]com",
      "AllowedPrefixes": [
        "example.com/project/package",
        "example.com/other/package"
      ],
      "ForbiddenPrefixes": [
        "example.com/legacy/package"
      ]
    },
    {
      "SelectorRegexp": "^unsafe$",
      "AllowedPrefixes": [],
      "ForbiddenPrefixes": [ "" ],
      "Transitive": true
    }
  ]
}
```

The `SelectorRegexp` specifies that this rule applies only to imports which
match that regex.

Note: an empty list (`[]`) matches nothing, and an empty string (`""`) is a
prefix of everything.

### What are InverseRules?

In contrast to rules, which are defined in terms of "things this package
depends on", inverse rules are defined in terms of "things which import this
package".  This allows for fine-grained import restrictions for "semi-private
packages" which are more sophisticated than Go's `internal` convention.

If inverse rules are found, then all known imports of the package are checked
against each such rule, in the same fashion as regular rules.  Note that this
can only handle known imports, which is defined as any package which is also
being considered by this `import-boss` run.  For most repositories, `./...` will
suffice.

Example:

```yaml
inverseRules:
  - selectorRegexp: example[.]com
    allowedPrefixes:
      - example.com/this-same-repo
      - example.com/close-friend/legacy
    forbiddenPrefixes:
      - example.com/other-project
  - selectorRegexp: example[.]com
    transitive: true
    forbiddenPrefixes:
      - example.com/other-team
```

## How do I run import-boss?

For most scenarios, simply running `import-boss ./...` will work.  For projects
which use Go workspaces, this can even span multiple modules.
