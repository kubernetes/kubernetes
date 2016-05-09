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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/devel/kubectl-conventions.md).

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
    - [Create commands](#create-commands)
  - [Flag conventions](#flag-conventions)
  - [Output conventions](#output-conventions)
  - [Documentation conventions](#documentation-conventions)
  - [Command implementation conventions](#command-implementation-conventions)
  - [Generators](#generators)

<!-- END MUNGE: GENERATED_TOC -->

## Principles

* Strive for consistency across commands
* Explicit should always override implicit
  * Environment variables should override default values
  * Command-line flags should override default values and environment variables
    * `--namespace` should also override the value specified in a specified resource

## Command conventions

* Command names are all lowercase, and hyphenated if multiple words.
* kubectl VERB NOUNs for commands that apply to multiple resource types.
* Command itself should not have built-in aliases.
* NOUNs may be specified as `TYPE name1 name2` or `TYPE/name1 TYPE/name2` or `TYPE1,TYPE2,TYPE3/name1`; TYPE is omitted when only a single type is expected.
* Resource types are all lowercase, with no hyphens; both singular and plural forms are accepted.
* NOUNs may also be specified by one or more file arguments: `-f file1 -f file2 ...`
* Resource types may have 2- or 3-letter aliases.
* Business logic should be decoupled from the command framework, so that it can be reused independently of kubectl, cobra, etc.
  * Ideally, commonly needed functionality would be implemented server-side in order to avoid problems typical of "fat" clients and to make it readily available to non-Go clients.
* Commands that generate resources, such as `run` or `expose`, should obey specific conventions, see [generators](#generators).
* A command group (e.g., `kubectl config`) may be used to group related non-standard commands, such as custom generators, mutations, and computations.


### Create commands

`kubectl create <resource>` commands fill the gap between "I want to try Kubernetes, but I don't know or care what gets created" (`kubectl run`) and "I want to create exactly this" (author yaml and run `kubectl create -f`).
They provide an easy way to create a valid object without having to know the vagaries of particular kinds, nested fields, and object key typos that are ignored by the yaml/json parser.
Because editing an already created object is easier than authoring one from scratch, these commands only need to have enough parameters to create a valid object and set common immutable fields.  It should default as much as is reasonably possible.
Once that valid object is created, it can be further manipulated using `kubectl edit` or the eventual `kubectl set` commands.

`kubectl create <resource> <special-case>` commands help in cases where you need to perform non-trivial configuration generation/transformation tailored for a common use case.
`kubectl create secret` is a good example, there's a `generic` flavor with keys mapping to files, then there's a `docker-registry` flavor that is tailored for creating an image pull secret,
and there's a `tls` flavor for creating tls secrets.  You create these as separate commands to get distinct flags and separate help that is tailored for the particular usage.


## Flag conventions

* Flags are all lowercase, with words separated by hyphens
* Flag names and single-character aliases should have the same meaning across all commands
* Command-line flags corresponding to API fields should accept API enums exactly (e.g., `--restart=Always`)
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
  * `-w`: Watch (currently also used for `--www` in `proxy`, but should be deprecated)
  * `-p`: Previous
    * also used for `--pod` in `exec`, but deprecated
    * also used for `--patch` in `patch`, but should be deprecated
    * also used for `--port` in `proxy`, but should be deprecated
  * `-P`: Static file prefix in `proxy`, but should be deprecated
  * `-r`: Replicas
  * `-u`: Unix socket
  * `-v`: Verbose logging level
* `--dry-run`: Don't modify the live state; simulate the mutation and display the output. All mutations should support it.
* `--local`: Don't contact the server; just do local read, transformation, generation, etc., and display the output
* `--output-version=...`: Convert the output to a different API group/version
* `--validate`: Validate the resource schema

## Output conventions

* By default, output is intended for humans rather than programs
  * However, affordances are made for simple parsing of `get` output
* Only errors should be directed to stderr
* `get` commands should output one row per resource, and one resource per row
  * Column titles and values should not contain spaces in order to facilitate commands that break lines into fields: cut, awk, etc. Instead, use `-` as the word separator.
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
* For fields that can be explicitly unset (booleans, integers, structs), the output should say `<unset>`.  Likewise, for arrays `<none>` should be used.  Lastly `<unknown>` should be used where unrecognized field type was specified.
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

## Command implementation conventions

For every command there should be a `NewCmd<CommandName>` function that creates the command and returns a pointer to a `cobra.Command`, which can later be added to other parent commands to compose the structure tree. There should also be a `<CommandName>Config` struct with a variable to every flag and argument declared by the command (and any other variable required for the command to run). This makes tests and mocking easier. The struct ideally exposes three methods:

* `Complete`: Completes the struct fields with values that may or may not be directly provided by the user, for example, by flags pointers, by the `args` slice, by using the Factory, etc.
* `Validate`: performs validation on the struct fields and returns appropriate errors.
* `Run<CommandName>`: runs the actual logic of the command, taking as assumption that the struct is complete with all required values to run, and they are valid.

Sample command skeleton:

```go
// MineRecommendedName is the recommended command name for kubectl mine.
const MineRecommendedName = "mine"

// MineConfig contains all the options for running the mine cli command.
type MineConfig struct {
  mineLatest bool
}

const (
  mineLong = `Some long description
for my command.`

  mineExample = `  # Run my command's first action
  $ %[1]s first

  # Run my command's second action on latest stuff
  $ %[1]s second --latest`
)

// NewCmdMine implements the kubectl mine command.
func NewCmdMine(parent, name string, f *cmdutil.Factory, out io.Writer) *cobra.Command {
  opts := &MineConfig{}

  cmd := &cobra.Command{
    Use:     fmt.Sprintf("%s [--latest]", name),
    Short:   "Run my command",
    Long:    mineLong,
    Example: fmt.Sprintf(mineExample, parent+" "+name),
    Run: func(cmd *cobra.Command, args []string) {
      if err := opts.Complete(f, cmd, args, out); err != nil {
        cmdutil.CheckErr(err)
      }
      if err := opts.Validate(); err != nil {
        cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
      }
      if err := opts.RunMine(); err != nil {
        cmdutil.CheckErr(err)
      }
    },
  }

  cmd.Flags().BoolVar(&options.mineLatest, "latest", false, "Use latest stuff")
  return cmd
}

// Complete completes all the required options for mine.
func (o *MineConfig) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
  return nil
}

// Validate validates all the required options for mine.
func (o MineConfig) Validate() error {
  return nil
}

// RunMine implements all the necessary functionality for mine.
func (o MineConfig) RunMine() error {
  return nil
}
```

The `Run<CommandName>` method should contain the business logic of the command and as noted in [command conventions](#command-conventions), ideally that logic should exist server-side so any client could take advantage of it. Notice that this is not a mandatory structure and not every command is implemented this way, but this is a nice convention so try to be compliant with it. As an example, have a look at how [kubectl logs](../../pkg/kubectl/cmd/logs.go) is implemented.

## Generators

Generators are kubectl commands that generate resources based on a set of inputs (other resources, flags, or a combination of both).

The point of generators is:
* to enable users using kubectl in a scripted fashion to pin to a particular behavior which may change in the future. Explicit use of a generator will always guarantee that the expected behavior stays the same.
* to enable potential expansion of the generated resources for scenarios other than just creation, similar to how -f is supported for most general-purpose commands.

Generator commands shoud obey to the following conventions:
* A `--generator` flag should be defined. Users then can choose between different generators, if the command supports them (for example, `kubectl run` currently supports generators for pods, jobs, replication controllers, and deployments), or between different versions of a generator so that users depending on a specific behavior may pin to that version (for example, `kubectl expose` currently supports two different versions of a service generator).
* Generation should be decoupled from creation. A generator should implement the `kubectl.StructuredGenerator` interface and have no dependencies on cobra or the Factory. See, for example, how the first version of the namespace generator is defined:

```go
// NamespaceGeneratorV1 supports stable generation of a namespace
type NamespaceGeneratorV1 struct {
  // Name of namespace
  Name string
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &NamespaceGeneratorV1{}

// StructuredGenerate outputs a namespace object using the configured fields
func (g *NamespaceGeneratorV1) StructuredGenerate() (runtime.Object, error) {
  if err := g.validate(); err != nil {
    return nil, err
  }
  namespace := &api.Namespace{}
  namespace.Name = g.Name
  return namespace, nil
}

// validate validates required fields are set to support structured generation
func (g *NamespaceGeneratorV1) validate() error {
  if len(g.Name) == 0 {
    return fmt.Errorf("name must be specified")
  }
  return nil
}
```

The generator struct (`NamespaceGeneratorV1`) holds the necessary fields for namespace generation. It also satisfies the `kubectl.StructuredGenerator` interface by implementing the `StructuredGenerate() (runtime.Object, error)` method which configures the generated namespace that callers of the generator (`kubectl create namespace` in our case) need to create.
* `--dry-run` should output the resource that would be created, without creating it.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/kubectl-conventions.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
