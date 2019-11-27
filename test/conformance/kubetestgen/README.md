# Kubetestgen

kubetestgen generates a list of behaviors for a resource based on the OpenAPI schema. The purpose is to bootstrap a list of behaviors, and not to produce the final list of behaviors. We expect that the resulting files will be curated to identify a meaningful set of behaviors for the conformance requirements of the targeted resource. This may include addition, modification, and removal of behaviors from the generated list.

**Example usage for PodSpec:**

```
bazel build //test/conformance/kubetestgen:kubetestgen
/bazel-out/k8-fastbuild/bin/test/conformance/kubetestgen/linux_amd64_stripped/kubetestgen --resource io.k8s.api.core.v1.PodSpec --area pod --schema api/openapi-spec/swagger.json --dir test/conformance/behaviors/
```

**Flags:**

- `schema` - a URL or local file name pointing to the JSON OpenAPI schema
- `resource` - the specific OpenAPI definition for which to generate behaviors
- `area` - the name to use for the area
- `dir` - the path to the behaviors directory (default current directory)

**Note**: The tool automatically generates suites based on the object type for a field. All primitive data types are grouped into a default suite, while object data types are grouped into their own suite, one per object.
