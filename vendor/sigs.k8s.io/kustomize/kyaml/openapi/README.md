# Built-in OpenAPI data

Kustomize embeds a compiled OpenAPI bundle for Kubernetes built-in types. The
runtime artifact is:

```
kubernetesapi/data/kubernetes-openapi-union-v1.21.2.bundle-v1.json.gz
```

The Kubernetes suffix identifies the newest Kubernetes schema represented by
the bundle. `bundle-v1` is the independent artifact format version. The
initial compiler migration uses a single v1.21.2 source, so both the coverage
floor and ceiling are v1.21.2.

The bundle contains the complete OpenAPI definitions and a compact index from
GVK to root definition and resource scope. API paths and other top-level
OpenAPI fields are compiler inputs and are not embedded in the runtime binary.

## Regenerating the bundle

The checked-in source is the gzip-compressed v1.21.2 OpenAPI protobuf at:

```
kubernetesapi/v1_21_2/swagger.pb.gz
```

Its uncompressed SHA-256 is:

```
5d171b55e9601912807a870d73ffe70bb306f5889a00e76986042a0f2d7b6bc2
```

Source acquisition is deliberately separate from compilation. When updating
Kubernetes, obtain the protobuf from an API server running the exact release,
review its provenance, and use the compiler's `-legacy-proto-output` flag to
write the deterministic checked-in `.pb.gz` archive. Update the embedded path,
version constants, and generated bundle together. Every source's uncompressed
digest is recorded in the bundle metadata.

Regenerate the runtime bundle with:

```
make -C kyaml/openapi generate
```

The compiler performs the protobuf-to-OpenAPI conversion, constructs the GVK
and scope index, validates local references, writes canonical JSON, and uses a
deterministic gzip header. The generated artifact must be byte-for-byte
reproducible. Verify it and run the OpenAPI tests with:

```
make -C kyaml/openapi verify
```

The protobuf archive is retained only as the compiler input and to preserve
the legacy public asset API. The normal Kustomize runtime does not import that
compatibility package.

The small Kustomization schema remains as source JSON at
`kustomizationapi/swagger.json` and is embedded directly with `go:embed`.
