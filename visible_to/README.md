# Package Groups Used in Kubernetes Visibility Rules

## Background

`BUILD` rules define dependencies, answering the question: on what packages
does _foo_ depend?

The `BUILD` file in this package allows one to define _allowed_ reverse
dependencies, answering the question: given a package _foo_, what other specific
packages are allowed to depend on it?

This is done via visibility rules.

Visibility rules discourage unintended, spurious dependencies that blur code
boundaries, slow CICD queues and generally inhibit progress.

#### Facts

* A package is any directory that contains a `BUILD` file.

* A `package_group` is a `BUILD` file term that defines a set of packages for
  use in other rules.

* A visibility rule takes a list of package groups as its argument - or one of
  the pre-defined groups `//visibility:private` or `//visibility:public`.

* If no visibility is explicitly defined, a package is _private_ by default.

* Violations in visibility cause `make bazel-build` to fail, which in turn causes
  the submit queue to fail - that's the enforcement.

#### Why define all package groups meant for visibility here (in one file)?

 * Ease discovery of appropriate groups for use in a rule.
 * Ease reuse (inclusions) of common used groups.
 * Consistent style:
    * easy to read `//visible_to:math_library_CONSUMERS` rules,
    * call out bad dependencies for eventual removal.
 * Make it more obvious in code reviews when visibility is being
   modified.
 * One set of `OWNERS` to manage visibility.

Its also possible to define the `package_group` used in a visibility rule right
next to said rule, scattering group definitions around the repository,
inhibiting discover, reuse, consistency, etc.

## Rule Examples

#### Nobody outside this package can depend on me.

```
visibility = ["//visibility:private"],
```

Since this is the default, there's no reason to use this rule except as a means
to override, for some specific target, some broader, whole-package visibility
rule.

#### Anyone can depend on me (eschew this).

```
visibility = ["//visibility:public"],
```

#### Only some servers can depend on me.

Appropriate for, say, backend storage utilities.

```
visibility = ["//visible_to:server_foo","//visible_to:server_bar"].
```

#### Both some client and some server can see me.

Appropriate for shared API definition files and generated code:

```
visibility = ["//visible_to:client_foo,//visible_to:server_foo"],
```

## Handy commands

#### Quickly check for visibility violations
```
bazel build --check_visibility --nobuild \
    //cmd/... //pkg/... //federation/... //plugin/... \
    //third_party/... //examples/... //test/... //vendor/k8s.io/...
```

#### Who depends on `//foo/bar`?

To create a seed set for a package group, one can ask what packages currently
depend on (must currently be able to see) the target
`//foo/bar:go_default_library`?  The result is a starting point to reduce
visibility if so desired. It's also a time consuming query, since it has to
understand all deps.

```
bazel query "rdeps(...,//foo/bar:go_default_library)" | \
    grep go_default_library | \
    sed 's/\(.*\):go_default_library/ "\1",/'
```

#### What targets below `//foo/bar` are visible to anyone?

A means to look for things one missed when locking down `//foo/bar`.

```
bazel query "visible(...,//foo/bar/...)"
```

#### What targets below `//foo/bar` may `//yada/client` depend on?

...without violating visibility rules? A means to pinoint unexpected visibility.

```
bazel query "visible(//yada/yida,//foo/bar/...)" | more
```

#### What packages does kubectl need from kubernetes?

```
  bazel query "buildfiles(deps(cmd/kubectl:kubectl))" | \
      grep -v @bazel_tools | \
      grep -v @io_bazel_rules | \
      grep -v @io_kubernetes_build | \
      grep -v @local_config | \
      grep -v @local_jdk | \
      grep -v //visible_to: | \
      sed 's/:BUILD//' | \
      sort | uniq > ~/KUBECTL_BUILD.txt
```

or try

```
  bazel query --nohost_deps --noimplicit_deps \
        "kind('source file', deps(cmd/kubectl:kubectl))" | wc -
```


#### How does kubectl depend on pkg/util/parsers?

```
bazel query "somepath(cmd/kubectl:kubectl, pkg/util/parsers:go_default_library)"
```

 

