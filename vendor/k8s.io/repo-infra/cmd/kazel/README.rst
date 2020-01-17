kazel - a BUILD file generator for go and bazel
===============================================

Requirements:
#############

* Your project must be somewhat compatible with go tool because
  kazel uses go tool to parse your import tree.
* You must have a **GOPATH** and **GOROOT** setup and your project must
  be in the correct location in your **GOPATH**.
* Your ``./vendor`` directory may not contain ``BUILD`` files.

Usage:
######

1. Get kazel by running ``go get k8s.io/repo-infra/kazel``.

2. Create a ``.kazelcfg.json`` in the root of the repository. For the
   kazel repository, the ``.kazelcfg.json`` would look like:

  .. code-block:: json

   {
     "GoPrefix": "k8s.io/repo-infra",
     "SrcDirs": [
       "./kazel"
     ],
     "SkippedPaths": [
       ".*foobar(baz)?.*$"
     ]
   }

3. Run kazel:

  .. code-block:: bash

    $ kazel -root=$GOPATH/src/k8s.io/repo-infra

Defaults:
#########

* **SrcDirs** in ``.kazelcfg.json`` defaults to ``["./"]``
* ``-root`` option defaults to the current working directory

Automanagement:
###############

kazel reconciles rules that have the "**automanaged**" tag. If
you no longer want kazel to manage a rule, you can remove the
**automanaged** tag and kazel will no longer manage that rule.

kazel only manages srcs, deps, and library attributes of a
rule after initial creation so you can add and managed other
attributes like data and copts and kazel will respect your
changes.

kazel automatically formats all ``BUILD`` files in your repository
except for those matching **SkippedPaths**.

Adding "sources" rules:
#######################

If you set "**AddSourcesRules**": ``true`` in your ``.kazelcfg.json``,
kazel will create "**package-srcs**" and "**all-srcs**" rules in every
package.

The "**package-srcs**" rule is a glob matching all files in the
package recursively, but not any files owned by packages in
subdirectories.

The "**all-srcs**" rule includes both the "**package-srcs**" rule and
the "**all-srcs**" rules of all subpackages; i.e. **//:all-srcs** will
include all files in your repository.

The "**package-srcs**" rule defaults to private visibility,
since it is safer to depend on the "**all-srcs**" rule: if a
subpackage is added, the "**package-srcs**" rule will no longer
include those files.

You can remove the "**automanaged**" tag from the "**package-srcs**"
rule if you need to modify the glob (such as adding excludes).
It's recommended that you leave the "**all-srcs**" rule
automanaged.

Validating BUILD files in CI:
#############################

If you run kazel with ``--validate``, it will not update any ``BUILD`` files, but it
will exit nonzero if any ``BUILD`` files are out-of-date. You can add ``--print-diff``
to print out the changes needed.
