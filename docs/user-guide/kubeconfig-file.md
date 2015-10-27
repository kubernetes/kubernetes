<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# kubeconfig files

Authentication in kubernetes can differ for different individuals.

- A running kubelet might have one way of authenticating (i.e. certificates).
- Users might have a different way of authenticating (i.e. tokens).
- Administrators might have a list of certificates which they provide individual users.
- There may be multiple clusters, and we may want to define them all in one place - giving users the ability to use their own certificates and reusing the same global configuration.

So in order to easily switch between multiple clusters, for multiple users, a kubeconfig file was defined.

This file contains a series of authentication mechanisms and cluster connection information associated with nicknames.  It also introduces the concept of a tuple of authentication information (user) and cluster connection information called a context that is also associated with a nickname.

Multiple kubeconfig files are allowed.  At runtime they are loaded and merged together along with override options specified from the command line (see rules below).

## Related discussion

http://issue.k8s.io/1755

## Example kubeconfig file

The below file contains a `current-context` which will be used by default by clients which are using the file to connect to a cluster.  Thus, this kubeconfig file has more information in it then we will necessarily have to use in a given session.  You can see it defines many clusters, and users associated with those clusters.  The context itself is associated with both a cluster AND a user.

```yaml
current-context: federal-context
apiVersion: v1
clusters:
- cluster:
    api-version: v1
    server: http://cow.org:8080
  name: cow-cluster
- cluster:
    certificate-authority: path/to/my/cafile
    server: https://horse.org:4443
  name: horse-cluster
- cluster:
    insecure-skip-tls-verify: true
    server: https://pig.org:443
  name: pig-cluster
contexts:
- context:
    cluster: horse-cluster
    namespace: chisel-ns
    user: green-user
  name: federal-context
- context:
    cluster: pig-cluster
    namespace: saw-ns
    user: black-user
  name: queen-anne-context
kind: Config
preferences:
  colors: true
users:
- name: blue-user
  user:
    token: blue-token
- name: green-user
  user:
    client-certificate: path/to/my/client/cert
    client-key: path/to/my/client/key
```

### Building your own kubeconfig file

NOTE, that if you are deploying k8s via kube-up.sh, you do not need to create your own kubeconfig files, the script will do it for you.

In any case, you can easily use this file as a template to create your own kubeconfig files.

So, lets do a quick walk through the basics of the above file so you can easily modify it as needed...

The above file would likely correspond to an api-server which was launched using the `--token-auth-file=tokens.csv` option, where the tokens.csv file looked something like this:

```
blue-user,blue-user,1
mister-red,mister-red,2
```

Also, since we have other users who validate using **other** mechanisms, the api-server would have probably been launched with other authentication options (there are many such options, make sure you understand which ones YOU care about before crafting a kubeconfig file, as nobody needs to implement all the different permutations of possible authentication schemes).

- Since the user for the current context is "green-user", any client of the api-server using this kubeconfig file would naturally be able to log in succesfully, because we are providigin the green-user's client credentials.
- Similarly, we can operate as the "blue-user" if we choose to change the value of current-context.

In the above scenario, green-user would have to log in by providing certificates, whereas blue-user would just provide the token.  All this information would be handled for us by the

## Loading and merging rules

The rules for loading and merging the kubeconfig files are straightforward, but there are a lot of them.  The final config is built in this order:
  1.  Get the kubeconfig  from disk.  This is done with the following hierarchy and merge rules:


      If the CommandLineLocation (the value of the `kubeconfig` command line option) is set, use this file only.  No merging.  Only one instance of this flag is allowed.


      Else, if EnvVarLocation (the value of $KUBECONFIG) is available, use it as a list of files that should be merged.
      Merge files together based on the following rules.
      Empty filenames are ignored.  Files with non-deserializable content produced errors.
      The first file to set a particular value or map key wins and the value or map key is never changed.
      This means that the first file to set CurrentContext will have its context preserved.  It also means that if two files specify a "red-user", only values from the first file's red-user are used.  Even non-conflicting entries from the second file's "red-user" are discarded.


      Otherwise, use HomeDirectoryLocation (~/.kube/config) with no merging.
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
  1.  Determine the actual user info to use. User is built using the same rules as cluster info, EXCEPT that you can only have one authentication technique per user.
      1. Load precedence is 1) command line flag, 2) user fields from kubeconfig
      1. The command line flags are: `client-certificate`, `client-key`, `username`, `password`, and `token`.
      1. If there are two conflicting techniques, fail.
  1.  For any information still missing, use default values and potentially prompt for authentication information

## Manipulation of kubeconfig via `kubectl config <subcommand>`

In order to more easily manipulate kubeconfig files, there are a series of subcommands to `kubectl config` to help.
See [kubectl/kubectl_config.md](kubectl/kubectl_config.md) for help.

### Example

```console
$ kubectl config set-credentials myself --username=admin --password=secret
$ kubectl config set-cluster local-server --server=http://localhost:8080
$ kubectl config set-context default-context --cluster=local-server --user=myself
$ kubectl config use-context default-context
$ kubectl config set contexts.default-context.namespace the-right-prefix
$ kubectl config view
```

produces this output

```yaml
apiVersion: v1
clusters:
- cluster:
    server: http://localhost:8080
  name: local-server
contexts:
- context:
    cluster: local-server
    namespace: the-right-prefix
    user: myself
  name: default-context
current-context: default-context
kind: Config
preferences: {}
users:
- name: myself
  user:
    password: secret
    username: admin
```

and a kubeconfig file that looks like this

```yaml
apiVersion: v1
clusters:
- cluster:
    server: http://localhost:8080
  name: local-server
contexts:
- context:
    cluster: local-server
    namespace: the-right-prefix
    user: myself
  name: default-context
current-context: default-context
kind: Config
preferences: {}
users:
- name: myself
  user:
    password: secret
    username: admin
```

#### Commands for the example file

```console
$ kubectl config set preferences.colors true
$ kubectl config set-cluster cow-cluster --server=http://cow.org:8080 --api-version=v1
$ kubectl config set-cluster horse-cluster --server=https://horse.org:4443 --certificate-authority=path/to/my/cafile
$ kubectl config set-cluster pig-cluster --server=https://pig.org:443 --insecure-skip-tls-verify=true
$ kubectl config set-credentials blue-user --token=blue-token
$ kubectl config set-credentials green-user --client-certificate=path/to/my/client/cert --client-key=path/to/my/client/key
$ kubectl config set-context queen-anne-context --cluster=pig-cluster --user=black-user --namespace=saw-ns
$ kubectl config set-context federal-context --cluster=horse-cluster --user=green-user --namespace=chisel-ns
$ kubectl config use-context federal-context
```

### Final notes for tying it all together

So, tying this all together, a quick start to creating your own kubeconfig file:

- Take a good look and understand how you're api-server is being launched: You need to know YOUR security requirements and policies before you can design a kubeconfig file for convenient authentication.

- Replace the snippet above with information for your cluster's api-server endpoint.

- Make sure your api-server is launched in such a way that at least one user (i.e. green-user) credentials are provided to it.  You will of course have to look at api-server documentation in order to determine the current state-of-the-art in terms of providing authentication details.






<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubeconfig-file.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
