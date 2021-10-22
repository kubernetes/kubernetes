# Kube OpenAPI Integration Tests

## Running the integration tests

Within the current directory:

```bash
$ go test -v .
```

## Generating the golden Swagger definition file and API rule violation report

First, run the generator to create `openapi_generated.go` file which specifies
the `OpenAPIDefinition` for each type, and generate the golden API rule
violation report file. Note that if you do not pass a report
filename (`./testdata/golden.v2.report` in the command below) to let the generator
to print API rule violations to the file, the generator will return error to stderr
on API rule violations.

```bash
$ go run ../../cmd/openapi-gen/openapi-gen.go \
  -i "k8s.io/kube-openapi/test/integration/testdata/custom,k8s.io/kube-openapi/test/integration/testdata/listtype,k8s.io/kube-openapi/test/integration/testdata/maptype,k8s.io/kube-openapi/test/integration/testdata/structtype,k8s.io/kube-openapi/test/integration/testdata/dummytype,k8s.io/kube-openapi/test/integration/testdata/uniontype,k8s.io/kube-openapi/test/integration/testdata/defaults" \
  -o pkg \
  -p generated \
  -O openapi_generated \
  -h ../../boilerplate/boilerplate.go.txt \
  -r ./testdata/golden.v2.report
```
The generated file `pkg/generated/openapi_generated.go` should have been created.

Next, run the OpenAPI builder to create the Swagger file which includes
the definitions. The output file named `golden.v2.json` will be output in
the current directory.

---
**NOTE:**

If you've created a new type, make sure you add it in `createWebServices()` in
`./builder/main.go`, or the definitions won't be generated.
---

---
**NOTE:**

If you've created a new package, make sure you also add it to to the
`inputDir` in `integration_suite_test.go`.
---

```bash
$ go run builder/main.go testdata/golden.v2.json
```
