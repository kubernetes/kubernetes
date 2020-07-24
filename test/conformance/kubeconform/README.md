# kubeconform

`kubeconform` is used to manage the creation and coverage analysis of conformance behaviors and tests. Currently it performs two functions:

* `gen`. This command generates a list of behaviors for a resource based on the OpenAPI schema. The purpose is to bootstrap a list of behaviors, and not to produce the final list of behaviors. We expect that the resulting files will be curated to identify a meaningful set of behaviors for the conformance requirements of the targeted resource. This may include addition, modification, and removal of behaviors from the generated list.
* `link`. This command prints the defined behaviors not covered by any test.

## gen
**Example usage for PodSpec:**

From the root directory of the k/k repo, will produce `pod.yaml` in
`test/conformance/behaviors`. The `pwd` is needed because of how bazel handles
working directories with `run`.

```
$ bazel run  //test/conformance/kubeconform:kubeconform -- --resource io.k8s.api.core.v1.PodSpec --area pod --schema api/openapi-spec/swagger.json --dir `pwd`/test/conformance/behaviors/ gen
```

**Flags:**

- `schema` - a URL or local file name pointing to the JSON OpenAPI schema
- `resource` - the specific OpenAPI definition for which to generate behaviors
- `area` - the name to use for the area
- `dir` - the path to the behaviors directory (default current directory)

**Note**: The tool automatically generates suites based on the object type for a field. All primitive data types are grouped into a default suite, while object data types are grouped into their own suite, one per object.

## link

```
$ bazel run  //test/conformance/kubeconform:kubeconform -- -dir `pwd`/test/conformance/behaviors/sig-node  -testdata `pwd`/test/conformance/testdata/conformance.yaml link
```
