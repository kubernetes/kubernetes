<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/kubectl-conventions.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Kubectl Conventions
===================

Updated: 8/27/2015

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Principles](#principles)
  - [Command conventions](#command-conventions)
  - [Flag conventions](#flag-conventions)
  - [Output conventions](#output-conventions)
  - [Documentation conventions](#documentation-conventions)

<!-- END MUNGE: GENERATED_TOC -->

## Principles

* Strive for consistency across commands
* Explicit should always override implicit
  * Environment variables should override default values
  * Command-line flags should override default values and environment variables
    * --namespace should also override the value specified in a specified resource

## Command conventions

* Command names are all lowercase, and hyphenated if multiple words.
* kubectl VERB NOUNs for commands that apply to multiple resource types
* NOUNs may be specified as TYPE name1 name2 ... or TYPE/name1 TYPE/name2; TYPE is omitted when only a single type is expected
* Resource types are all lowercase, with no hyphens; both singular and plural forms are accepted
* NOUNs may also be specified by one or more file arguments: -f file1 -f file2 ...
* Resource types may have 2- or 3-letter aliases.
* Business logic should be decoupled from the command framework, so that it can be reused independently of kubectl, cobra, etc.
  * Ideally, commonly needed functionality would be implemented server-side in order to avoid problems typical of "fat" clients and to make it readily available to non-Go clients
* Commands that generate resources, such as `run` or `expose`, should obey the following conventions:
  * Flags should be converted to a parameter Go map or json map prior to invoking the generator
  * The generator must be versioned so that users depending on a specific behavior may pin to that version, via `--generator=`
  * Generation should be decoupled from creation
  * `--dry-run` should output the resource that would be created, without creating it
* A command group (e.g., `kubectl config`) may be used to group related non-standard commands, such as custom generators, mutations, and computations

## Flag conventions

* Flags are all lowercase, with words separated by hyphens
* Flag names and single-character aliases should have the same meaning across all commands
* Command-line flags corresponding to API fields should accept API enums exactly (e.g., --restart=Always)
* Do not reuse flags for different semantic purposes, and do not use different flag names for the same semantic purpose -- grep for `"Flags()"` before adding a new flag
* Use short flags sparingly, only for the most frequently used options, prefer lowercase over uppercase for the most common cases, try to stick to well known conventions for UNIX commands and/or Docker, where they exist, and update this list when adding new short flags
  * `-f`: Resource file
    * also used for `--follow` in `logs`, but should be deprecated in favor of `-F`
  * `-l`: Label selector
    * also used for `--labels` in `expose`, but should be deprecated
  * `-L`: Label columns
  * `-c`: Container
    * also used for `--client` in `version`, but should be deprecated
  * `-i`: Attach stdin
  * `-t`: Allocate TTY
    * also used for `--template`, but deprecated
  * `-w`: Watch (currently also used for `--www` in `proxy`, but should be deprecated)
  * `-p`: Previous
    * also used for `--pod` in `exec`, but deprecated
    * also used for `--patch` in `patch`, but should be deprecated
    * also used for `--port` in `proxy`, but should be deprecated
  * `-P`: Static file prefix in `proxy`, but should be deprecated
  * `-r`: Replicas
  * `-u`: Unix socket
  * `-v`: Verbose logging level
* `--dry-run`: Don't modify the live state; simulate the mutation and display the output
* `--local`: Don't contact the server; just do local read, transformation, generation, etc. and display the output
* `--output-version=...`: Convert the output to a different API group/version
* `--validate`: Validate the resource schema

## Output conventions

* By default, output is intended for humans rather than programs
  * However, affordances are made for simple parsing of `get` output
* Only errors should be directed to stderr
* `get` commands should output one row per resource, and one resource per row
  * Column titles and values should not contain spaces in order to facilitate commands that break lines into fields: cut, awk, etc.
  * By default, `get` output should fit within about 80 columns
    * Eventually we could perhaps auto-detect width
    * `-o wide` may be used to display additional columns
  * The first column should be the resource name, titled `NAME` (may change this to an abbreviation of resource type)
  * NAMESPACE should be displayed as the first column when --all-namespaces is specified
  * The last default column should be time since creation, titled `AGE`
  * `-Lkey` should append a column containing the value of label with key `key`, with `<none>` if not present
  * json, yaml, Go template, and jsonpath template formats should be supported and encouraged for subsequent processing
    * Users should use --api-version or --output-version to ensure the output uses the version they expect
* `describe` commands may output on multiple lines and may include information from related resources, such as events. Describe should add additional information from related resources that a normal user may need to know - if a user would always run "describe resource1" and the immediately want to run a "get type2" or "describe resource2", consider including that info. Examples, persistent volume claims for pods that reference claims, events for most resources, nodes and the pods scheduled on them. When fetching related resources, a targeted field selector should be used in favor of client side filtering of related resources.
* Mutations should output TYPE/name verbed by default, where TYPE is singular; `-o name` may be used to just display TYPE/name, which may be used to specify resources in other commands

## Documentation conventions

* Commands are documented using Cobra; docs are then auto-generated by `hack/update-generated-docs.sh`.
  * Use should contain a short usage string for the most common use case(s), not an exhaustive specification
  * Short should contain a one-line explanation of what the command does
  * Long may contain multiple lines, including additional information about input, output, commonly used flags, etc.
  * Example should contain examples
    * Start commands with `$`
    * A comment should precede each example command, and should begin with `#`
* Use "FILENAME" for filenames
* Use "TYPE" for the particular flavor of resource type accepted by kubectl, rather than "RESOURCE" or "KIND"
* Use "NAME" for resource names

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/kubectl-conventions.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
