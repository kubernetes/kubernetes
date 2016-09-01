<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Generation and release cycle of clientset

Client-gen is an automatic tool that generates
[clientset](../../docs/proposals/client-package-structure.md#high-level-client-sets)
based on API types. This doc introduces the use the client-gen, and the release
cycle of the generated clientsets.

## Using client-gen

The workflow includes four steps:
- Marking API types with tags: in `pkg/apis/${GROUP}/${VERSION}/types.go`, mark
the types (e.g., Pods) that you want to generate clients for with the
`// +genclient=true` tag. If the resource associated with the type is not
namespace scoped (e.g., PersistentVolume), you need to append the
`nonNamespaced=true` tag as well.

- Running the client-gen tool: you need to use the command line argument
`--input` to specify the groups and versions of the APIs you want to generate
clients for, client-gen will then look into
`pkg/apis/${GROUP}/${VERSION}/types.go` and generate clients for the types you
have marked with the `genclient` tags. For example, running:

```
$ client-gen --input="api/v1,extensions/v1beta1" --clientset-name="my_release"
```

will generate a clientset named "my_release" which includes clients for api/v1
objects and extensions/v1beta1 objects. You can run `$ client-gen --help` to see
other command line arguments.

- ***Adding expansion methods***: client-gen only generates the common methods,
  such as `Create()` and `Delete()`. You can manually add additional methods
  through the expansion interface. For example, this
  [file](../../pkg/client/clientset_generated/release_1_4/typed/core/v1/pod_expansion.go)
  adds additional methods to Pod's client. As a convention, we put the expansion
  interface and its methods in file ${TYPE}_expansion.go. In most cases, you
  don't want to remove existing expansion files. So to make life easier,
  instead of creating a new clientset from scratch, ***you can copy and rename an
  existing clientset (so that all the expansion files are copied)***, and then run
  client-gen.

- Generating fake clients for testing purposes: client-gen will generate a fake
clientset if the command line argument `--fake-clientset` is set. The fake
clientset provides the default implementation, you only need to fake out the
methods you care about when writing test cases.

The output of client-gen includes:

- clientset: the clientset will be generated at
`pkg/client/clientset_generated/` by default, and you can change the path via
the `--clientset-path` command line argument.

- Individual typed clients and client for group: They will be generated at `pkg/client/clientset_generated/${clientset_name}/typed/generated/${GROUP}/${VERSION}/`

## Released clientsets

At the 1.2 release, we have two released clientsets in the repo:
internalclientset and release_1_2.

- internalclientset: because most components in our repo still deal with the
internal objects, the internalclientset talks in internal objects to ease the
adoption of clientset. We will keep updating it as our API evolves. Eventually
it will be replaced by a versioned clientset.

- release_1_2: release_1_2 clientset is a versioned clientset, it includes
clients for the core v1 objects, extensions/v1beta1, autoscaling/v1, and
batch/v1 objects. We will NOT update it after we cut the 1.2 release. After the
1.2 release, we will create release_1_3 clientset and keep it updated until we
cut release 1.3.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/generating-clientset.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
