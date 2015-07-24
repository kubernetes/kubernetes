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
[here](http://releases.k8s.io/release-1.0/docs/design/expansion.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Variable expansion in pod command, args, and env

## Abstract

A proposal for the expansion of environment variables using a simple `$(var)` syntax.

## Motivation

It is extremely common for users to need to compose environment variables or pass arguments to
their commands using the values of environment variables.  Kubernetes should provide a facility for
the 80% cases in order to decrease coupling and the use of workarounds.

## Goals

1.  Define the syntax format
2.  Define the scoping and ordering of substitutions
3.  Define the behavior for unmatched variables
4.  Define the behavior for unexpected/malformed input

## Constraints and Assumptions

*  This design should describe the simplest possible syntax to accomplish the use-cases
*  Expansion syntax will not support more complicated shell-like behaviors such as default values
   (viz: `$(VARIABLE_NAME:"default")`), inline substitution, etc.

## Use Cases

1.  As a user, I want to compose new environment variables for a container using a substitution
    syntax to reference other variables in the container's environment and service environment
    variables
1.  As a user, I want to substitute environment variables into a container's command
1.  As a user, I want to do the above without requiring the container's image to have a shell
1.  As a user, I want to be able to specify a default value for a service variable which may
    not exist
1.  As a user, I want to see an event associated with the pod if an expansion fails (ie, references
    variable names that cannot be expanded)

### Use Case: Composition of environment variables

Currently, containers are injected with docker-style environment variables for the services in
their pod's namespace.  There are several variables for each service, but users routinely need
to compose URLs based on these variables because there is not a variable for the exact format
they need. Users should be able to build new environment variables with the exact format they need.
Eventually, it should also be possible to turn off the automatic injection of the docker-style
variables into pods and let the users consume the exact information they need via the downward API
and composition.

#### Expanding expanded variables

It should be possible to reference an variable which is itself the result of an expansion, if the
referenced variable is declared in the container's environment prior to the one referencing it.
Put another way -- a container's environment is expanded in order, and expanded variables are
available to subsequent expansions.

### Use Case: Variable expansion in command

Users frequently need to pass the values of environment variables to a container's command.
Currently, Kubernetes does not perform any expansion of variables.  The workaround is to invoke a
shell in the container's command and have the shell perform the substitution, or to write a wrapper
script that sets up the environment and runs the command.  This has a number of drawbacks:

1.  Solutions that require a shell are unfriendly to images that do not contain a shell
2.  Wrapper scripts make it harder to use images as base images
3.  Wrapper scripts increase coupling to Kubernetes

Users should be able to do the 80% case of variable expansion in command without writing a wrapper
script or adding a shell invocation to their containers' commands.

### Use Case: Images without shells

The current workaround for variable expansion in a container's command requires the container's
image to have a shell.  This is unfriendly to images that do not contain a shell (`scratch` images,
for example).  Users should be able to perform the other use-cases in this design without regard to
the content of their images.

### Use Case: See an event for incomplete expansions

It is possible that a container with incorrect variable values or command line may continue to run
for a long period of time, and that the end-user would have no visual or obvious warning of the
incorrect configuration.  If the kubelet creates an event when an expansion references a variable
that cannot be expanded, it will help users quickly detect problems with expansions.

## Design Considerations

### What features should be supported?

In order to limit complexity, we want to provide the right amount of functionality so that the 80%
cases can be realized and nothing more.  We felt that the essentials boiled down to:

1.  Ability to perform direct expansion of variables in a string
2.  Ability to specify default values via a prioritized mapping function but without support for
    defaults as a syntax-level feature

### What should the syntax be?

The exact syntax for variable expansion has a large impact on how users perceive and relate to the
feature.  We considered implementing a very restrictive subset of the shell `${var}` syntax.  This
syntax is an attractive option on some level, because many people are familiar with it.  However,
this syntax also has a large number of lesser known features such as the ability to provide
default values for unset variables, perform inline substitution, etc.

In the interest of preventing conflation of the expansion feature in Kubernetes with the shell
feature, we chose a different syntax similar to the one in Makefiles, `$(var)`.  We also chose not
to support the bar `$var` format, since it is not required to implement the required use-cases.

Nested references, ie, variable expansion within variable names, are not supported.

#### How should unmatched references be treated?

Ideally, it should be extremely clear when a variable reference couldn't be expanded.  We decided
the best experience for unmatched variable references would be to have the entire reference, syntax
included, show up in the output.  As an example, if the reference `$(VARIABLE_NAME)` cannot be
expanded, then `$(VARIABLE_NAME)` should be present in the output.

#### Escaping the operator

Although the `$(var)` syntax does overlap with the `$(command)` form of command substitution
supported by many shells, because unexpanded variables are present verbatim in the output, we
expect this will not present a problem to many users.  If there is a collision between a variable
name and command substitution syntax, the syntax can be escaped with the form `$$(VARIABLE_NAME)`,
which will evaluate to `$(VARIABLE_NAME)` whether `VARIABLE_NAME` can be expanded or not.

## Design

This design encompasses the variable expansion syntax and specification and the changes needed to
incorporate the expansion feature into the container's environment and command.

### Syntax and expansion mechanics

This section describes the expansion syntax, evaluation of variable values, and how unexpected or
malformed inputs are handled.

#### Syntax

The inputs to the expansion feature are:

1.  A utf-8 string (the input string) which may contain variable references
2.  A function (the mapping function) that maps the name of a variable to the variable's value, of
    type `func(string) string`

Variable references in the input string are indicated exclusively with the syntax
`$(<variable-name>)`.  The syntax tokens are:

- `$`: the operator
- `(`: the reference opener
- `)`: the reference closer

The operator has no meaning unless accompanied by the reference opener and closer tokens.  The
operator can be escaped using `$$`.  One literal `$` will be emitted for each `$$` in the input.

The reference opener and closer characters have no meaning when not part of a variable reference.
If a variable reference is malformed, viz: `$(VARIABLE_NAME` without a closing expression, the
operator and expression opening characters are treated as ordinary characters without special
meanings.

#### Scope and ordering of substitutions

The scope in which variable references are expanded is defined by the mapping function.  Within the
mapping function, any arbitrary strategy may be used to determine the value of a variable name.
The most basic implementation of a mapping function is to use a `map[string]string` to lookup the
value of a variable.

In order to support default values for variables like service variables presented by the kubelet,
which may not be bound because the service that provides them does not yet exist, there should be a
mapping function that uses a list of `map[string]string` like:

```go
func MakeMappingFunc(maps ...map[string]string) func(string) string {
	return func(input string) string {
		for _, context := range maps {
			val, ok := context[input]
			if ok {
				return val
			}
		}

		return ""
    }
}

// elsewhere
containerEnv := map[string]string{
	"FOO":           "BAR",
	"ZOO":           "ZAB",
	"SERVICE2_HOST": "some-host",
}

serviceEnv := map[string]string{
	"SERVICE_HOST": "another-host",
	"SERVICE_PORT": "8083",
}

// single-map variation
mapping := MakeMappingFunc(containerEnv)

// default variables not found in serviceEnv
mappingWithDefaults := MakeMappingFunc(serviceEnv, containerEnv)
```

### Implementation changes

The necessary changes to implement this functionality are:

1.  Add a new interface, `ObjectEventRecorder`, which is like the `EventRecorder` interface, but
    scoped to a single object, and a function that returns an `ObjectEventRecorder` given an
    `ObjectReference` and an `EventRecorder`
2.  Introduce `third_party/golang/expansion` package that provides:
    1.  An `Expand(string, func(string) string) string` function
    2.  A `MappingFuncFor(ObjectEventRecorder, ...map[string]string) string` function
3.  Make the kubelet expand environment correctly
4.  Make the kubelet expand command correctly

#### Event Recording

In order to provide an event when an expansion references undefined variables, the mapping function
must be able to create an event.  In order to facilitate this, we should create a new interface in
the `api/client/record` package which is similar to `EventRecorder`, but scoped to a single object:

```go
// ObjectEventRecorder knows how to record events about a single object.
type ObjectEventRecorder interface {
	// Event constructs an event from the given information and puts it in the queue for sending.
	// 'reason' is the reason this event is generated. 'reason' should be short and unique; it will
	// be used to automate handling of events, so imagine people writing switch statements to
	// handle them. You want to make that easy.
	// 'message' is intended to be human readable.
	//
	// The resulting event will be created in the same namespace as the reference object.
	Event(reason, message string)

	// Eventf is just like Event, but with Sprintf for the message field.
	Eventf(reason, messageFmt string, args ...interface{})

	// PastEventf is just like Eventf, but with an option to specify the event's 'timestamp' field.
	PastEventf(timestamp util.Time, reason, messageFmt string, args ...interface{})
}
```

There should also be a function that can construct an `ObjectEventRecorder` from a `runtime.Object`
and an `EventRecorder`:

```go
type objectRecorderImpl struct {
	object   runtime.Object
	recorder EventRecorder
}

func (r *objectRecorderImpl) Event(reason, message string) {
	r.recorder.Event(r.object, reason, message)
}

func ObjectEventRecorderFor(object runtime.Object, recorder EventRecorder) ObjectEventRecorder {
	return &objectRecorderImpl{object, recorder}	
}
```

#### Expansion package

The expansion package should provide two methods:

```go
// MappingFuncFor returns a mapping function for use with Expand that
// implements the expansion semantics defined in the expansion spec; it
// returns the input string wrapped in the expansion syntax if no mapping
// for the input is found.  If no expansion is found for a key, an event
// is raised on the given recorder.
func MappingFuncFor(recorder record.ObjectEventRecorder, context ...map[string]string) func(string) string {
	// ...
}

// Expand replaces variable references in the input string according to
// the expansion spec using the given mapping function to resolve the
// values of variables.
func Expand(input string, mapping func(string) string) string {
	// ...
}
```

#### Kubelet changes

The Kubelet should be made to correctly expand variables references in a container's environment,
command, and args.  Changes will need to be made to:

1.  The `makeEnvironmentVariables` function in the kubelet; this is used by
    `GenerateRunContainerOptions`, which is used by both the docker and rkt container runtimes
2.  The docker manager `setEntrypointAndCommand` func has to be changed to perform variable
    expansion
3.  The rkt runtime should be made to support expansion in command and args when support for it is
    implemented

### Examples

#### Inputs and outputs

These examples are in the context of the mapping:

| Name        | Value      |
|-------------|------------|
| `VAR_A`     | `"A"`      |
| `VAR_B`     | `"B"`      |
| `VAR_C`     | `"C"`      |
| `VAR_REF`   | `$(VAR_A)` |
| `VAR_EMPTY` |  `""`      |

No other variables are defined.

| Input                          | Result                     |
|--------------------------------|----------------------------|
| `"$(VAR_A)"`                   | `"A"`                      |
| `"___$(VAR_B)___"`             | `"___B___"`                |
| `"___$(VAR_C)"`                | `"___C"`                   |
| `"$(VAR_A)-$(VAR_A)"`          | `"A-A"`                    |
| `"$(VAR_A)-1"`                 | `"A-1"`                    |
| `"$(VAR_A)_$(VAR_B)_$(VAR_C)"` | `"A_B_C"`                  |
| `"$$(VAR_B)_$(VAR_A)"`         | `"$(VAR_B)_A"`             |
| `"$$(VAR_A)_$$(VAR_B)"`        | `"$(VAR_A)_$(VAR_B)"`      |
| `"f000-$$VAR_A"`               | `"f000-$VAR_A"`            |
| `"foo\\$(VAR_C)bar"`           | `"foo\Cbar"`               |
| `"foo\\\\$(VAR_C)bar"`         | `"foo\\Cbar"`              |
| `"foo\\\\\\\\$(VAR_A)bar"`     | `"foo\\\\Abar"`            |
| `"$(VAR_A$(VAR_B))"`           | `"$(VAR_A$(VAR_B))"`       |
| `"$(VAR_A$(VAR_B)"`            | `"$(VAR_A$(VAR_B)"`        |
| `"$(VAR_REF)"`                 | `"$(VAR_A)"`               |
| `"%%$(VAR_REF)--$(VAR_REF)%%"` | `"%%$(VAR_A)--$(VAR_A)%%"` |
| `"foo$(VAR_EMPTY)bar"`         | `"foobar"`                 |
| `"foo$(VAR_Awhoops!"`          | `"foo$(VAR_Awhoops!"`      |
| `"f00__(VAR_A)__"`             | `"f00__(VAR_A)__"`         |
| `"$?_boo_$!"`                  | `"$?_boo_$!"`              |
| `"$VAR_A"`                     | `"$VAR_A"`                 |
| `"$(VAR_DNE)"`                 | `"$(VAR_DNE)"`             |
| `"$$$$$$(BIG_MONEY)"`          | `"$$$(BIG_MONEY)"`         |
| `"$$$$$$(VAR_A)"`              | `"$$$(VAR_A)"`             |
| `"$$$$$$$(GOOD_ODDS)"`         | `"$$$$(GOOD_ODDS)"`        |
| `"$$$$$$$(VAR_A)"`             | `"$$$A"`                   |
| `"$VAR_A)"`                    | `"$VAR_A)"`                |
| `"${VAR_A}"`                   | `"${VAR_A}"`               |
| `"$(VAR_B)_______$(A"`         | `"B_______$(A"`            |
| `"$(VAR_C)_______$("`          | `"C_______$("`             |
| `"$(VAR_A)foobarzab$"`         | `"Afoobarzab$"`            |
| `"foo-\\$(VAR_A"`              | `"foo-\$(VAR_A"`           |
| `"--$($($($($--"`              | `"--$($($($($--"`          |
| `"$($($($($--foo$("`           | `"$($($($($--foo$("`       |
| `"foo0--$($($($("`             | `"foo0--$($($($("`         |
| `"$(foo$$var)`                 | `$(foo$$var)`              |

#### In a pod: building a URL

Notice the `$(var)` syntax.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: expansion-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "env" ]
      env:
        - name: PUBLIC_URL
          value: "http://$(GITSERVER_SERVICE_HOST):$(GITSERVER_SERVICE_PORT)"
  restartPolicy: Never
```

#### In a pod: building a URL using downward API

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: expansion-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "env" ]
      env:
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: "metadata.namespace"
        - name: PUBLIC_URL
          value: "http://gitserver.$(POD_NAMESPACE):$(SERVICE_PORT)"
  restartPolicy: Never
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/expansion.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
