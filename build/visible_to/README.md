# Package Groups Used in Kubernetes Visibility Rules

## Background

`BUILD` rules define dependencies, answering the question:
on what packages does _foo_ depend?

The `BUILD` file in this package allows one to define
_allowed_ reverse dependencies, answering the question:
given a package _foo_, what other specific packages are
allowed to depend on it?

This is done via visibility rules.

Visibility rules discourage unintended, spurious
dependencies that blur code boundaries, slow CICD queues and
generally inhibit progress.

#### Facts

* A package is any directory that contains a `BUILD` file.

* A `package_group` is a `BUILD` file rule that defines a named
  set of packages for use in other rules, e.g., given
  ```
  package_group(
    name = "database_CONSUMERS",
    packages = [
        "//foo/dbinitializer",
        "//foo/backend/...",  # `backend` and everything below it
    ],
  )
  ```
  one can specify the following visibility rule in any `BUILD` rule:
  ```
  visibility = [ "//build/visible_to:database_CONSUMERS" ],
  ```

* A visibility rule takes a list of package groups as its
  argument - or one of the pre-defined groups
  `//visibility:private` or `//visibility:public`.

* If no visibility is explicitly defined, a package is
  _private_ by default.

* Violations in visibility cause `make bazel-build` to fail,
  which in turn causes the submit queue to fail - that's the
  enforcement.

#### Why define all package groups meant for visibility here (in one file)?

 * Ease discovery of appropriate groups for use in a rule.
 * Ease reuse (inclusions) of commonly used groups.
 * Consistent style:
    * easy to read `//build/visible_to:math_library_CONSUMERS` rules,
    * call out bad dependencies for eventual removal.
 * Make it more obvious in code reviews when visibility is being
   modified.
 * One set of `OWNERS` to manage visibility.

The alternative is to use special [package literals] directly
in visibility rules, e.g.

```
  visibility = [
        "//foo/dbinitializer:__pkg__",
        "//foo/backend:__subpackages__",
  ],
```

The difference in style is similar to the difference between
using a named static constant like `MAX_NODES` rather than a
literal like `12`.  Names are preferable to literals for intent
documentation, search, changing one place rather than _n_,
associating usage in distant code blocks, etc.


## Rule Examples

#### Nobody outside this package can depend on me.

```
visibility = ["//visibility:private"],
```

Since this is the default, there's no reason to use this
rule except as a means to override, for some specific
target, some broader, whole-package visibility rule.

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
    //cmd/... //pkg/... //plugin/... \
    //third_party/... //examples/... //test/... //vendor/k8s.io/...
```

#### Who depends on target _q_?

To create a seed set for a visibility group, one can ask what
packages currently depend on (must currently be able to see) a
given Go library target?  It's a time consuming query.

```
q=//pkg/kubectl/cmd:go_default_library
bazel query "rdeps(...,${q})" | \
    grep go_default_library | \
    sed 's/\(.*\):go_default_library/ "\1",/'
```

#### What targets below _p_ are visible to anyone?

A means to look for things one missed when locking down _p_.

```
p=//pkg/kubectl/cmd
bazel query "visible(...,${p}/...)"
```

#### What packages below _p_ may target _q_ depend on without violating visibility rules?

A means to pinpoint unexpected visibility.

```
p=//pkg/kubectl
q=//cmd/kubelet:kubelet
bazel query "visible(${q},${p}/...)" | more
```

#### What packages does target _q_ need?

```
q=//cmd/kubectl:kubectl
bazel query "buildfiles(deps($q))" | \
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
    "kind('source file', deps($q))" | wc -
```


#### How does kubectl depend on pkg/util/parsers?

```
bazel query "somepath(cmd/kubectl:kubectl, pkg/util/parsers:go_default_library)"
```



[package literals]: https://bazel.build/versions/master/docs/be/common-definitions.html#common.visibility
