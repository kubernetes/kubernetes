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
[here](http://releases.k8s.io/release-1.0/docs/proposals/api-types-package-structure.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# API Types package layout

We have written tooling around api types and will write more. This document
seeks to prescribe an organizational layout for types packages and their
associated generated files.

Explicit goals are:
* Organizing our chaos.
* Make the toolchain easy to understand and use.
* Reduce the package import graph to a minimum for any given use case.
* Make it possible to set up a Kubernetes-style types directory in an arbitrary
  location; decouple our tools and github repository.
* Coordinate among those who will or might write autogeneration.
* Where does validation code go? How is it called?
* Where does conversion code go? How is it called?
* Where does defaulting code go? How is it called?
* Where are validation/conversion/defaulting functions registered?

Non-goals for this document:
* apiserver usage of these types.
* apiserver swagger generation for types.
* removing the "internal" version.
* Where does deep-copy code go? How is it called? (Deep copy is more general
  than API types.)

These will be covered in another document.

## Current state

Currently, we have api types in two places, `pkg/api` and `pkg/apis/`. The
former has many things that are general to all apis and needs to be split, but I
include everything for completeness.

```
pkg/api/
├── context.go
├── context_test.go
├── conversion.go
├── conversion_test.go
├── copy_test.go
├── deep_copy_generated.go
├── deep_copy_test.go
├── doc.go
├── endpoints
│   ├── util.go
│   └── util_test.go
├── errors
│   ├── doc.go
│   ├── errors.go
│   ├── errors_test.go
│   └── etcd
│       ├── doc.go
│       └── etcd.go
├── generate.go
├── generate_test.go
├── helpers.go
├── helpers_test.go
├── install
│   ├── install.go
│   └── install_test.go
├── latest
│   ├── doc.go
│   ├── latest.go
│   └── latest_test.go
├── mapper.go
├── meta
│   ├── doc.go
│   ├── interfaces.go
│   ├── meta.go
│   ├── meta_test.go
│   ├── restmapper.go
│   └── restmapper_test.go
├── meta.go
├── meta_test.go
├── node_example.json
├── pod_example.json
├── ref.go
├── ref_test.go
├── registered
│   └── registered.go
├── register.go
├── replication_controller_example.json
├── requestcontext.go
├── resource
│   ├── quantity_example_test.go
│   ├── quantity.go
│   ├── quantity_test.go
│   └── suffix.go
├── resource_helpers.go
├── resource_helpers_test.go
├── rest
│   ├── create.go
│   ├── create_test.go
│   ├── delete.go
│   ├── doc.go
│   ├── rest.go
│   ├── resttest
│   │   └── resttest.go
│   ├── types.go
│   ├── update.go
│   └── update_test.go
├── serialization_test.go
├── testapi
│   ├── testapi.go
│   └── testapi_test.go
├── testing
│   ├── compat
│   │   └── compatibility_tester.go
│   ├── fuzzer.go
│   └── pod_specs.go
├── types.generated.go
├── types.go
├── unversioned
│   ├── duration.go
│   ├── duration_test.go
│   ├── time.go
│   ├── time_test.go
│   ├── types.go
│   └── types_swagger_doc_generated.go
├── util
│   ├── group_version.go
│   └── group_version_test.go
├── v1
│   ├── backward_compatibility_test.go
│   ├── conversion_generated.go
│   ├── conversion.go
│   ├── conversion_test.go
│   ├── deep_copy_generated.go
│   ├── defaults.go
│   ├── defaults_test.go
│   ├── doc.go
│   ├── register.go
│   ├── types.generated.go
│   ├── types.go
│   └── types_swagger_doc_generated.go
└── validation
    ├── doc.go
    ├── events.go
    ├── events_test.go
    ├── schema.go
    ├── schema_test.go
    ├── testdata
    │   └── v1
    │       ├── invalidPod1.json
    │       ├── invalidPod2.json
    │       ├── invalidPod3.json
    │       ├── invalidPod.yaml
    │       └── validPod.yaml
    ├── validation.go
    └── validation_test.go

```

The api in `pkg/apis` is closer to where we want to be.

```
pkg/apis/
└── extensions
    ├── deep_copy_generated.go
    ├── helpers.go
    ├── helpers_test.go
    ├── install
    │   ├── install.go
    │   └── install_test.go
    ├── register.go
    ├── types.generated.go
    ├── types.go
    ├── v1beta1
    │   ├── conversion_generated.go
    │   ├── conversion.go
    │   ├── deep_copy_generated.go
    │   ├── defaults.go
    │   ├── defaults_test.go
    │   ├── register.go
    │   ├── types.generated.go
    │   ├── types.go
    │   └── types_swagger_doc_generated.go
    └── validation
        ├── validation.go
        └── validation_test.go
```

## Desired directory structure

We want to define a standard layout and structure for Kubernetes API types.
Below is a first draft, with explanation afterwards. We may try to move the
relevant parts of the legacy `pkg/api/v1` api into the new structure, although
this api doesn't have a group.

```
apis
├── group
│   ├── install
│   │   └── install_all_versions.go
│   ├── resource1
│   │   └── types.go
│   ├── resource2
│   │   └── types.go
│   ├── v1
│   │   ├── install
│   │   │   └── install_all_resources.go
│   │   ├── resource1
│   │   │   ├── conversion
│   │   │   │   ├── generated_conversion.go
│   │   │   │   └── manual_conversion_overrides.go
│   │   │   ├── defaulting
│   │   │   │   ├── generated_defaulters.go
│   │   │   │   └── generated_default_stubs.go
│   │   │   ├── types.go
│   │   │   └── validation
│   │   │       ├── generated_validation.go
│   │   │       └── manual_validation_overrides.go
│   │   └── resource2
│   │       ├── conversion
│   │       │   ├── generated_conversion.go
│   │       │   └── manual_conversion_overrides.go
│   │       ├── defaulting
│   │       │   ├── generated_defaulters.go
│   │       │   └── generated_default_stubs.go
│   │       ├── types.go
│   │       └── validation
│   │           ├── generated_validation.go
│   │           └── manual_validation_overrides.go
│   ├── v1beta1
│   │   ├── install
│   │   │   └── install_all_resources.go
│   │   ├── resource1
│   │   │   ├── conversion
│   │   │   │   ├── generated_conversion.go
│   │   │   │   └── manual_conversion_overrides.go
│   │   │   ├── defaulting
│   │   │   │   ├── generated_defaulters.go
│   │   │   │   └── generated_default_stubs.go
│   │   │   ├── types.go
│   │   │   └── validation
│   │   │       ├── generated_validation.go
│   │   │       └── manual_validation_overrides.go
│   │   └── resource2
│   │       ├── conversion
│   │       │   ├── generated_conversion.go
│   │       │   └── manual_conversion_overrides.go
│   │       ├── defaulting
│   │       │   ├── generated_defaulters.go
│   │       │   └── generated_default_stubs.go
│   │       ├── types.go
│   │       └── validation
│   │           ├── generated_validation.go
│   │           └── manual_validation_overrides.go
│   └── v2alpha1
│       ├── install
│       │   └── install_all_resources.go
│       ├── resource1
│       │   ├── conversion
│       │   │   ├── generated_conversion.go
│       │   │   └── manual_conversion_overrides.go
│       │   ├── defaulting
│       │   │   ├── generated_defaulters.go
│       │   │   └── generated_default_stubs.go
│       │   ├── types.go
│       │   └── validation
│       │       ├── generated_validation.go
│       │       └── manual_validation_overrides.go
│       ├── resource2
│       │   ├── conversion
│       │   │   ├── generated_conversion.go
│       │   │   └── manual_conversion_overrides.go
│       │   ├── defaulting
│       │   │   ├── generated_defaulters.go
│       │   │   └── generated_default_stubs.go
│       │   ├── types.go
│       │   └── validation
│       │       ├── generated_validation.go
│       │       └── manual_validation_overrides.go
│       └── resource3
│           ├── conversion
│           │   ├── generated_conversion.go
│           │   └── manual_conversion_overrides.go
│           ├── defaulting
│           │   ├── generated_defaulters.go
│           │   └── generated_default_stubs.go
│           ├── types.go
│           └── validation
│               ├── generated_validation.go
│               └── manual_validation_overrides.go
└── unversioned
    ├── meta
    │   └── meta.go
    ├── resource
    │   └── quantity.go
    ├── time
    │   └── time.go
    └── types.go
```

### Package: `apis/`

Multiple groups may live in the same tree. Each group gets its own directory
under `apis/`. We'll assume an `apis/` directory of this structure in all tools.
In the above example, we show one group, named 'group'.

TBD: Either test code will live here, or no code.

### Package: `apis/unversioned/`

This package and sub-packages are allowed to be imported and used by any
group/version/resource. There is no internal/versioned distinction. Types in
this directory in the main kubernetes repository are reusable by anyone in any
API; creators of 3rd party APIs should think carefully before creating their own
unversioned package.

### Package: `apis/group/`

Multiple versions live in a group. Each version gets its own subdirectory.

Additionally, a directory for each resource has a types.go file containing an
"internal" version of the group's objects. These files are used as a
destination/source for conversion functions.

### Package: `apis/group/install`

This package installs *all* of the versions of the group, meaning it imports
each nested install/ package.

### Package: `apis/group/v1`

'v1' is taken to be the current stable version of this api.

Multiple resource directories (we show 'resource1' and 'resource2') each have a
`types.go`, containing the type definitions for this group/version/resource.

This package has tightly controlled imports. It's allowed to include the
Kubernetes unversioned API types (to be moved in/under `pkg/apis/unversioned`)
and little else. In particular, it *must not* import its parent `apis/group`.

### Package: `apis/group/v1/install`

This package, if imported, registers (via an `init()`) the v1 version of every
resource in this group, its conversion, validation, and defaulting functions.

We have separate install packages for every version to allow deliberate import
choices to be made.

### Package: `apis/group/v1/resource1/conversion`

This package contains conversion functions, and exports a Register() function
which will register them (but does not register them as a side effect of being
imported).

There are two sorts of conversion functions: those which have been automatically
generated, and those which have manual overrides. All the conversion functions
convert to & from the types in the parent `apis/group/resource1` and
`apis/group/v1/resource1` directories.

The allowed imports are constrained; in particular, nothing that would let you
do RPCs to look up other cluster state is allowed.

### Package: `apis/group/v1/resource1/defaulting`

This package contains defaulting functions, and exports a Register() function
which will register them (but does not register them as a side effect of being
imported).

The functions apply to only the `group/v1/resource1` types, *not* the parent
`group/resource1` types.

High-level requirements:
* Any tags that autogeneration pays attention to (e.g., `// +default: xxxx`)
  should be readable by humans as well as machines.
* Things that require a default should not compile until the defaulting function
  is supplied.
* Defaults may need to be hirearchical-- e.g., the container resources in a pod
  temaplate are defaulted differently than the resources in a pod. The simplest
  way to do this is to set the defaults in the function for the template and
  separately for the pod, instead of setting them on the container's resources.

The allowed imports are constrained; in particular, nothing that would let you
do RPCs to look up other cluster state is allowed. Defaulting that requires
information not contained in the object is not allowed; you must change your
design until it's not needed.

### Package: `apis/group/v1/resource1/validation`

This package contains validation functions, and if imported, will register them.

The functions apply to only the `group/v1/resource1` types, *not* the parent
`group/resource1` types.

The allowed imports are constrained; in particular, nothing that would let you
do RPCs to look up other cluster state is allowed. Validation functions validate
*only* the object, not cluster-wide or other constraints, which must be validate
elsewhere.

There are two sorts of validation functions, those that can be automatically
written and those that need human attention.

The former will be autogenerated by looking for directives in the comments
preceding the types or their embedded struct members. Allowed directives will
include:
* `// +validation=minmax{<numeric value>, <numeric value>}`
* `// +validation=regexp/<regexp>/`
* `// +validation=<exact value>`

If no validation directive is in the comments, a stub with a
`panic("write me!")` marker will be generated. To prevent this, a
`// +validation=none` directive may be added to the comment.

(We can hash out the exact form of the tags later; it should be easy for human
readers to figure out what will happen.)

### Package: `apis/group/v1beta1`

This package layout is the same as `apis/group/v1`, but it is an older version
of the API group, which may lack features.

### Package: `apis/group/v2alpha1`

This package layout is the same as `apis/group/v1`, but it is an newer version
of the API group, which may have extra features. It has `resource3`, for
example.

It is a rule that a client should never have to pay attention to multiple
versions of a group at the same time, so v2alpha1 copies all types from v1, even
if there's no changes.

## Type support package layout

The above is how to arrange the types for a particular API. But we also must
consider, what Kubernetes packages do you need to import to use the API types?

```
k8s.io/api_machinery/
└── typetools
    ├── convert
    ├── deep
    ├── default
    └── validate
```

These packages will live in the kubernetes repo (or another repo that the
kubernetes team will own and maintain). The interface for all the packages is
similar: there'll be a `Register(...)` function for the install packages to use,
and some function to perform the action. So, e.g., `convert.Convert(&in, &out)`,
`bar := deep.Copy(&foo)`, `default.Apply(&object)`,
`errs := validate.Check(&object)`.

More detailed descriptions belong in another doc when we hash out the
particulars of the code generation; the important thing here is just to know
that we'll supply packages with:
* Minimal import tree
* Some sort of registration method. Registration marks a thing as available and
  it can be enumerated (for producing lists of what could be turned on, for
  example).
* Some sort of enabling method, for actually turning a thing on. Registration
  and enabling are conflated in our codebase at the moment, and this makes it
  difficult to, for example, construct the default value for --runtime-config.
* Some sort of execution method, for actually using a thing.

## Reusable apiserver infrastructure packages

In the future, it'd be nice if we could generate REST install code based on
comment annotations in the go types. See #16560 for a problem statement.

## Desired toolchain

Along with packages for runtime use, we will supply tools to do the various
generations.

Ideally, the tool will take the root `apis/` directory, discover the `group`s
and `version`s, and perform the generations outlined above. We will generate as
much as possible. The goal is for you to write a `apis/group/types.go` file and
an `apis/group/v1alpha1` file, push a button, and get the rest of the package
structure, including tests that check and enforce the package imports.

We'll deliver a tool with few enough command line arguments that you won't feel
compelled to wrap it with a bash script. It will be callable via `go generate`.

The tool should never fail to run because *its own* output from a previous
invocation is missing or broken, but it may fail (with helpful error messages)
if the input files with user types/functions don't compile.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/api-types-package-structure.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
