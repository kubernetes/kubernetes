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
└── group
    ├── install
    ├── types.go
    ├── v1
    │   ├── install
    │   │   ├── conversion
    │   │   │   ├── generated_conversion.go
    │   │   │   └── manual_conversion_overrides.go
    │   │   ├── defaulting
    │   │   │   ├── generated_defaulters.go
    │   │   │   └── generated_default_stubs.go
    │   │   └── validation
    │   │       ├── generated_validation.go
    │   │       └── generated_validation_stubs.go
    │   └── types.go
    ├── v1beta1
    │   ├── install
    │   │   ├── conversion
    │   │   │   ├── generated_conversion.go
    │   │   │   └── manual_conversion_overrides.go
    │   │   ├── defaulting
    │   │   │   ├── generated_defaulters.go
    │   │   │   └── generated_default_stubs.go
    │   │   └── validation
    │   │       ├── generated_validation.go
    │   │       └── generated_validation_stubs.go
    │   └── types.go
    └── v2alpha1
        ├── install
        │   ├── conversion
        │   │   ├── generated_conversion.go
        │   │   └── manual_conversion_overrides.go
        │   ├── defaulting
        │   │   ├── generated_defaulters.go
        │   │   └── generated_default_stubs.go
        │   └── validation
        │       ├── generated_validation.go
        │       └── generated_validation_stubs.go
        └── types.go
```

### Package: `apis/`

Multiple groups may live in the same tree. Each group gets its own directory
under `apis/`. We'll assume an `apis/` directory of this structure in all tools.

TBD: Either test code will live here, or no code.

### Package: `apis/group/`

Multiple versions live in a group. Each version gets its own subdirectory.

Additionally, a types.go file contains an "internal" version of the group's
objects. This file is used as a destination/source for conversion functions.

### Package: `apis/group/install`

This package installs *all* of the versions of the group.

### Package: `apis/group/v1`

'v1' is taken to be the current stable version of this api.

`types.go` contains the types for this version of the api group.

This package has tightly controlled imports. It's allowed to include the
Kubernetes unversioned API types (to be moved to `pkg/apis/unversioned`) and
little else. In particular, it *must not* import its parent `apis/group`.

### Package: `apis/group/v1/install`

This package, if imported, registers the v1 version of this group, its
conversion, validation, and defaulting functions.

We have separate install packages for every version to allow deliberate import
choices to be made.

### Package: `apis/group/v1/install/conversion`

This package contains conversion functions, and if imported, will register them.

There are two sorts of conversion functions: those which could be automatically
generated, and those which have manual additions. All the conversion functions
convert to & from the types in the parent `apis/groups` and `apis/groups/v1`
directories.

### Package: `apis/group/v1/install/defaulting`

This package contains defaulting functions, and if imported, will register them.

The functions apply to only the `group/v1` types, *not* the parent `group/`
types.

There are two sorts of defaulting functions, those that can be automatically
written and those that need human attention.

The former will be autogenerated by looking for `// +default=<value>` comments
preceding the types or their embedded struct members. Similarly, a
`// +default=custom` marker will force a stub to be generated. Stubs will be
generated with a `panic("write me!")` marker to ensure that they are populated.

### Package: `apis/group/v1/install/validation`

This package contains validation functions, and if imported, will register them.

The functions apply to only the `group/v1` types, *not* the parent `group/`
types.

The allowed imports are constrained; in particular, nothing that would let you
do RPCs to look up other cluster state is allowed. Validation functions validate
*only* the object, not cluster-wide or other constraints, which must be validate
elsewhere. (TODO: copy-pasta this 2x above)

There are two sorts of validation functions, those that can be automatically
written and those that need human attention.

The former will be autogenerated by looking for directives in the comments
preceding the types or their embedded struct members. Allowed directives will
include:
* `// +validate_min=<numeric value>,validate_max=<numeric value>`
* `// +validate_regexp=<regexp>`
* `// +validate_exact=<exact value>`

If no validation directive is in the comments, a stub with a
`panic("write me!")` marker will be generated. To prevent this, a
`// +no_validation_required` directive may be added to the comment.

### Package: `apis/group/v1beta1`

This package layout is the same as `apis/group/v1`, but it is an older version
of the API group, which may lack features.

### Package: `apis/group/v2alpha1`

This package layout is the same as `apis/group/v1`, but it is an newer version
of the API group, which may have extra features.

## Type support package layout

The above is how to arrange the types for a particular API. But we also must
consider, what Kubernetes packages do you need to import to use the API types?

```
pkg
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
* Some sort of registration method
* Some sort of execution method

## Reusable apiserver infrastructure packages

WIP

In the future, it'd be nice if we could generate REST install code based on
comment annotations in the go types.

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

