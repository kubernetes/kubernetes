# Fork of github.com/gogo/protobuf v1.3.2

This directory holds the parts of `github.com/gogo/protobuf` v1.3.2 that the
`protoc-gen-gogo` plugin in `cmd/go-to-protobuf` needs. It exists so that
Kubernetes does not depend on the unmaintained upstream module at generation
time (KEP-5589, phase two). The license is in `LICENSE`.

Kept from upstream, with Go import paths rewritten to this directory:

- `gogoproto` (with `gogo.proto`)
- `proto`
- `protoc-gen-gogo/descriptor`, `protoc-gen-gogo/generator`,
  `protoc-gen-gogo/generator/internal/remap`, `protoc-gen-gogo/plugin`
- `plugin/marshalto`, `plugin/size`, `plugin/stringer`, `plugin/unmarshal`
- `vanity`, `vanity/command`

Changes from upstream:

- Import declarations point at this directory. Import paths that the
  generator writes into generated code (`github.com/gogo/protobuf/proto`,
  `.../sortkeys`, `.../types`) are unchanged; `go-to-protobuf` rewrites or
  drops them after generation.
- `vanity/command` registers only the four plugins above and no longer runs
  the `testgen` plugin.
- `gogoproto/helper.go` imports the descriptor package without the
  `google_protobuf` alias; `hack/verify-pkg-names.sh` rejects underscores in
  import aliases.
- `protoc-gen-gogo/descriptor/descriptor_gostring.gen.go`,
  `plugin/size/sizetest.go`, `plugin/stringer/stringertest.go`, Makefiles,
  golden files and tests are not copied.

Do not edit generated `.pb.go` files here by hand.
