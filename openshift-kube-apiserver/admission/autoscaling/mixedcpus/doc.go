package mixedcpus

//The admission should provide the following functionalities:
//1. In case a user specifies more than a single `openshift.io/enable-shared-cpus` resource,
//it rejects the pod request with an error explaining the user how to fix its pod spec.
//2. It rejects a non-guaranteed pod which is asking for `openshift.io/enable-shared-cpus` resource.
//3. It adds an annotation `cpu-shared.crio.io` that will be used to tell the runtime that shared cpus were requested.
//For every container requested for shared cpus, it adds an annotation with the following scheme:
//`cpu-shared.crio.io/<container name>`
//4. It validates that the pod deployed in a namespace that has `workload.mixedcpus.openshift.io/allowed` annotation.
