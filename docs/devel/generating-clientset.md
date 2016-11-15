# Generation and release cycle of clientset

Client-gen is an automatic tool that generates [clientset](../../docs/proposals/client-package-structure.md#high-level-client-sets) based on API types. This doc introduces the use the client-gen, and the release cycle of the generated clientsets.

## Using client-gen

The workflow includes three steps:

1. Marking API types with tags: in `pkg/apis/${GROUP}/${VERSION}/types.go`, mark the types (e.g., Pods) that you want to generate clients for with the `// +genclient=true` tag. If the resource associated with the type is not namespace scoped (e.g., PersistentVolume), you need to append the `nonNamespaced=true` tag as well.

2.
  - a. If you are developing in the k8s.io/kubernetes repository, you just need to run hack/update-codegen.sh.

  - b. If you are running client-gen outside of k8s.io/kubernetes, you need to use the command line argument `--input` to specify the groups and versions of the APIs you want to generate clients for, client-gen will then look into `pkg/apis/${GROUP}/${VERSION}/types.go` and generate clients for the types you have marked with the `genclient` tags. For example, to generated a clientset named "my_release" including clients for api/v1 objects and extensions/v1beta1 objects, you need to run:

``` 
$ client-gen --input="api/v1,extensions/v1beta1" --clientset-name="my_release"
```

3. ***Adding expansion methods***: client-gen only generates the common methods, such as CRUD. You can manually add additional methods through the expansion interface. For example, this [file](../../pkg/client/clientset_generated/release_1_5/typed/core/v1/pod_expansion.go) adds additional methods to Pod's client. As a convention, we put the expansion interface and its methods in file ${TYPE}_expansion.go. In most cases, you don't want to remove existing expansion files. So to make life easier, instead of creating a new clientset from scratch, ***you can copy and rename an existing clientset (so that all the expansion files are copied)***, and then run client-gen.

## Output of client-gen

- clientset: the clientset will be generated at `pkg/client/clientset_generated/` by default, and you can change the path via the `--clientset-path` command line argument.

- Individual typed clients and client for group: They will be generated at `pkg/client/clientset_generated/${clientset_name}/typed/generated/${GROUP}/${VERSION}/`

## Released clientsets

If you are contributing code to k8s.io/kubernetes, try to use the release_X_Y clientset in this [directory](../../pkg/client/clientset_generated/).

If you need a stable Go client to build your own project, please refer to the [client-go repository](https://github.com/kubernetes/client-go).

We are migrating k8s.io/kubernetes to use client-go as well, see issue [#35159](https://github.com/kubernetes/kubernetes/issues/35159).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS --> [![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/generating-clientset.md?pixel)]() <!-- END MUNGE: GENERATED_ANALYTICS -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/generating-clientset.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
