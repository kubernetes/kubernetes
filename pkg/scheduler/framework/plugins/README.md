# Scheduler Framework Plugins

## Creating a new in-tree plugin

Read [the docs](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
to understand the different extension points within the scheduling framework.

TODO(#95156): flesh this out a bit more.

## Adding plugin configuration parameters through `KubeSchedulerConfiguration`

You can give users the ability to configure parameters in scheduler plugins using
[`KubeSchedulerConfiguration`](https://kubernetes.io/docs/reference/scheduling/config/).
This section covers how you can add arguments to existing in-tree plugins [(example PR)](https://github.com/kubernetes/kubernetes/pull/94814).
Let's assume the plugin is called `FooPlugin` and we want to add an optional
integer parameter named `barParam`.

### Defining and registering the struct

First, we need to define a struct type named `FooPluginArgs` in
`pkg/scheduler/apis/config/types_pluginargs.go`, which is the representation of
the configuration parameters that is internal to the scheduler.

```go
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type FooPluginArgs struct {
	// metav1 is k8s.io/apimachinery/pkg/apis/meta/v1
	metav1.TypeMeta
	BarParam int32
}
```

Note that we embed `k8s.io/apimachinery/pkg/apis/meta/v1.TypeMeta` to include
API metadata for [versioning and persistence](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#api-conventions).
We add the `+k8s:deepcopy-gen:interfaces` comment to [auto-generate a `DeepCopy` function](https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/code-generator)
for the struct.

Similarly, define `FooPluginArgs` in `k8s.io/kube-scheduler/config/{version}/types_pluginargs.go`,
which is the versioned representation used in the `kube-scheduler` binary used
for deserialization. This time, however, in order to allow implicit default
values for arguments, the type of the struct's fields may be pointers; leaving
a parameter unspecified will set the pointer field to its zero value (nil),
which can be used to let the framework know that it must fill in the default
value. `BarParam` is of type `int32` and let's say we want a non-zero default
value for it:

```go
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type FooPluginArgs struct {
	metav1.TypeMeta `json:",inline"`
	BarParam *int32 `json:"barParam,omitempty"`
}
```

For each `types_pluginargs.go` addition, remember to register the type in the
corresponding `register.go`, which will allow the scheduler to recognize
`KubeSchedulerConfiguration` values at parse-time.

### Setting defaults

When a `KubeSchedulerConfiguration` object is parsed (happens in
`cmd/kube-scheduler/app/options/options.go`), the scheduler will convert from
the versioned type to the internal type, filling in the unspecified fields with
defaults. Speaking of defaults, define `SetDefaults_FooPluginArgs` in
`pkg/scheduler/apis/config/v1beta1/defaults.go` as follows:

```go
// v1beta1 refers to k8s.io/kube-scheduler/config/v1beta1.
func SetDefaults_FooPluginArgs(obj *v1beta1.FooPluginArgs) {
	if obj.BarParam == nil {
		obj.BarParam = pointer.Int32Ptr(42)
	}
}
```

### Validating configuration at runtime

Next, we need to define validators to make sure the user's configuration and
your default values are valid. To do this, add something like this in
`pkg/scheduler/apis/config/validation/validation_pluginargs.go`:

```go
// From here on, FooPluginArgs refers to the type defined in pkg/scheduler
// definition, not the kube-scheduler definition. We're dealing with
// post-default values.
func ValidateFooPluginArgs(args config.FooPluginArgs) error {
	if args.BarParam < 0 && args.BarParam > 100 {
		return fmt.Errorf("must be in the range [0, 100]")
	}
	return nil
}
```

### Code generation

We have defined everything necessary to run code generation now. Remember to
commit all your changes (not sure why this is needed) and do a `make clean`
first. Then:

```sh
$ cd $GOPATH/src/k8s.io/kubernetes
$ git add -A && git commit
$ make clean
$ ./hack/update-codegen.sh
$ make generated_files
```

This should automatically generate code to deep copy objects, convert between
different struct types, convert pointer types to raw types, and set defaults.

### Testing

After code generation, go back and write tests for all of the changes you made
in the previous section:

- `pkg/scheduler/apis/config/v1beta1/defaults_test.go` to unit test the
  defaults.
- `pkg/scheduler/apis/config/validation/validation_pluginargs_test.go` to unit
  test the validator.
- `pkg/scheduler/apis/config/scheme/scheme_test.go` to test the whole pipeline
  using a `KubeSchedulerConfiguration` definition.

### Receiving the arguments in the plugin

We can now finally receive `FooPluginArgs` in the plugin code. To do this,
modify the plugin's `New` method signature like so:

```go
func New(fpArgs runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
	// config.FooPluginArgs refers to the pkg/scheduler struct type definition.
	args, ok := fpArgs.(*config.FooPluginArgs)
	if !ok {
		return nil, fmt.Errorf("got args of type %T, want *FooPluginArgs", fpArgs)
	}
	if err := validation.ValidateFooPluginArgs(*args); err != nil {
		return nil, err
	}
	// Use args.BarParam as you like.
}
```
