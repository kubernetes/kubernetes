# .kubeconfig files
In order to easily switch between multiple clusters, a .kubeconfig file was defined.  This file contains a series of authentication mechanisms and cluster connection information associated with nicknames.  It also introduces the concept of a tuple of authentication information (user) and cluster connection information called a context that is also associated with a nickname.

Multiple files are .kubeconfig files are allowed.  At runtime they are loaded and merged together along with override options specified from the command line (see rules below).

## Related discussion
https://github.com/GoogleCloudPlatform/kubernetes/issues/1755

## Example .kubeconfig file
```
preferences: 
  colors: true
clusters:
  cow-cluster:
    server: http://cow.org:8080
    api-version: v1beta1
  horse-cluster:
    server: https://horse.org:4443
    certificate-authority: path/to/my/cafile
  pig-cluster:
    server: https://pig.org:443
    insecure-skip-tls-verify: true
users:
  black-user:
    auth-path: path/to/my/existing/.kubernetes_auth file
  blue-user:
    token: blue-token
  green-user:
    client-certificate: path/to/my/client/cert
    client-key: path/to/my/client/key
contexts:
  queen-anne-context:
    cluster: pig-cluster
    user: black-user
    namespace: saw-ns
  federal-context:
    cluster: horse-cluster
    user: green-user
    namespace: chisel-ns
current-context: federal-context
```

## Loading and merging rules
The rules for loading and merging the .kubeconfig files are straightforward, but there are a lot of them.  The final config is built in this order:
  1.  Merge together the kubeconfig itself.  This is done with the following hierarchy and merge rules:
      
      Empty filenames are ignored.  Files with non-deserializable content produced errors.
      The first file to set a particular value or map key wins and the value or map key is never changed.
      This means that the first file to set CurrentContext will have its context preserved.  It also means that if two files specify a "red-user", only values from the first file's red-user are used.  Even non-conflicting entries from the second file's "red-user" are discarded.
      1.  CommandLineLocation - the value of the `kubeconfig` command line option
      1.  EnvVarLocation - the value of $KUBECONFIG
      1.  CurrentDirectoryLocation - ``pwd``/.kubeconfig
      1.  HomeDirectoryLocation = ~/.kube/.kubeconfig
  1.  Determine the context to use based on the first hit in this chain
      1.  command line argument - the value of the `context` command line option
      1.  current-context from the merged kubeconfig file
      1.  Empty is allowed at this stage
  1.  Determine the cluster info and user to use.  At this point, we may or may not have a context.  They are built based on the first hit in this chain.  (run it twice, once for user, once for cluster)
      1.  command line argument - `user` for user name and `cluster` for cluster name
      1.  If context is present, then use the context's value
      1.  Empty is allowed
  1.  Determine the actual cluster info to use.  At this point, we may or may not have a cluster info.  Build each piece of the cluster info based on the chain (first hit wins):
      1.  command line arguments - `server`, `api-version`, `certificate-authority`, and `insecure-skip-tls-verify`
      1.  If cluster info is present and a value for the attribute is present, use it.
      1.  If you don't have a server location, error.
  1.  User is build using the same rules as cluster info, EXCEPT that you can only have one authentication  technique per user.

      The command line flags are: `auth-path`, `client-certificate`, `client-key`, and `token`.  If there are two conflicting techniques, fail.
  1.  For any information still missing, use default values and potentially prompt for authentication information
