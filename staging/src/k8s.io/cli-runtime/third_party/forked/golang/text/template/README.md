# Forked `text/template`

`exec.go`, `funcs.go`, `helper.go`, `option.go`, `template.go` and `internal/fmtsort/sort.go`
are copied from the Go 1.27.1 standard library (BSD license in `../../LICENSE`);
`text/template/parse` is imported from the standard library unchanged. Verify with:

    diff -r "$(go env GOROOT)/src/text/template" staging/src/k8s.io/cli-runtime/third_party/forked/golang/text/template

One change: `evalField` in `exec.go` does not resolve a name to a method. The standard
package calls `reflect.Value.MethodByName` with the template's field name, and a reachable
call with a non-constant name makes the Go linker keep every exported method of every type
in the binary (`cmd/link/internal/ld/deadcode.go`, `reflectSeen`). kubectl and kubeadm shrink
by about 30 and 49 percent when nothing in them reaches such a call.

Templates executed with this package may use struct fields, map keys and template functions.
`{{ .Name }}` on a value with a `Name` method and no `Name` field is an error; register a
function instead. golang/go#72895 proposes the same behavior upstream as `Template.ExecuteLite`.
