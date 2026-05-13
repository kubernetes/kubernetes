This file documents Go API changes in apimachinery.

Breaking API changes *must* be documented here, together with instructions on
how to deal with them or why they are expected to have no impact.

Go API changes are typically not included in the Kubernetes release notes, so
non-breaking noteworthy Go API changes *may* be documented here if they are
useful to know about for developers.

### KEP-4222: Support CBOR encoding for non-resource endpoints.

See [PR #139632](https://github.com/kubernetes/kubernetes/pull/139632).

```
- ./pkg/runtime.UseNondeterministicEncoding: changed from func(Encoder) Encoder to func(Serializer) Encoder
```

### Introduce Deferred Gen concept to Validation-gen framework

See [PR #138205](https://github.com/kubernetes/kubernetes/pull/138205).

```
- ./pkg/api/validate.IfOption: changed from func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, string, bool, func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T) k8s.io/apimachinery/pkg/util/validation/field.ErrorList) k8s.io/apimachinery/pkg/util/validation/field.ErrorList to func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, T, T, string, bool, func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, T, T) k8s.io/apimachinery/pkg/util/validation/field.ErrorList) k8s.io/apimachinery/pkg/util/validation/field.ErrorList
```

### Revert DV native error matcher and errors

See [PR #137017](https://github.com/kubernetes/kubernetes/pull/137017).

```
- ./pkg/util/validation/field.(*Error).MarkDeclarativeNative: removed
- ./pkg/util/validation/field.Error.DeclarativeNative: removed
- ./pkg/util/validation/field.ErrorList.MarkDeclarativeNative: removed
- ./pkg/util/validation/field.ErrorMatcher.ByDeclarativeNative: removed
```

### KEP-4671: Add Declarative Validation to Workload API

See [PR #135164](https://github.com/kubernetes/kubernetes/pull/135164).

```
- ./pkg/api/validation/path.NameMayNotBe: removed
- ./pkg/api/validation/path.NameMayNotContain: removed
```

### KEP-5589 - drop gogo runtime dependencies

See [PR #134256](https://github.com/kubernetes/kubernetes/pull/134256).

```
- ./pkg/api/resource.(*Quantity).Descriptor: removed
- ./pkg/api/resource.(*Quantity).ProtoMessage: removed
- ./pkg/api/resource.(*Quantity).XXX_DiscardUnknown: removed
- ./pkg/api/resource.(*Quantity).XXX_Marshal: removed
- ./pkg/api/resource.(*Quantity).XXX_Merge: removed
- ./pkg/api/resource.(*Quantity).XXX_Size: removed
- ./pkg/api/resource.(*Quantity).XXX_Unmarshal: removed
...
- ./pkg/util/intstr.(*IntOrString).XXX_Unmarshal: removed
```

### feat(validation-gen): Add "cohorts" & Tighten and simplify test framework

See [PR #134347](https://github.com/kubernetes/kubernetes/pull/134347).

```
- ./pkg/util/validation/field.TestIntf.Logf: removed
```

### gogo protobuf dependency cleanup

See [PR #134228](https://github.com/kubernetes/kubernetes/pull/134228).

```
- ./pkg/runtime.ProtobufMarshaller.Size: added
- ./pkg/runtime.ProtobufReverseMarshaller.Size: added
```

### Add  +k8s:ifEnabled, +k8s:ifDisabled and +k8s:enumExclude tags

See [PR #133768](https://github.com/kubernetes/kubernetes/pull/133768).

```
- ./pkg/api/validate.Enum: changed from func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, k8s.io/apimachinery/pkg/util/sets.Set[T]) k8s.io/apimachinery/pkg/util/validation/field.ErrorList to func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, k8s.io/apimachinery/pkg/util/sets.Set[T], []EnumExclusion[T]) k8s.io/apimachinery/pkg/util/validation/field.ErrorList
```

### Changes for Kubernetes <= 1.34

For older changes refer to the commit messages and PR descriptions.
