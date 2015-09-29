<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/kubeconfig-file.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# kubeconfig files

Authentication across Kubernetes can differ, for example:

- A running kubelet might have one way of authenticating (i.e. certificates).
- Users might all have different ways of authenticating (i.e. tokens).
- Administrators might provide individual certificates to users.
- For multiple clusters, they might be all defined in a single file that provides users with the ability to use their own certificates.

In order to accommodate switching between multiple clusters and supporting the varying number of users and their different authentication methods, Kubernetes uses a kubeconfig file.

The *kubeconfig file* contains a series of authentication mechanisms for both users and clusters. The file contains tuples of user authentication and cluster connection information that the Kubernetes api-server uses to establish connections. Information for users and clusters are defined under the respective `users` and `clusters` sections. There is also a `contexts` section that defines nicknames for the associated namespaces, clusters, and users.

You can create and use multiple kubeconfig files in your clusters. At runtime the kubeconfig files are loaded and merged together using the override options that you specify. Loading and merging kubeconfig files provides you the option for specifying your configuration information in separate files or consolidating it all into a single kubeconfig file. Using a single file allows for simplified global configuration that all your clusters share. See the [Loading and merging rules](#loading-and-merging-rules) below for more information.

#### Table of contents:

- [Example kubeconfig file](#example-kubeconfig-file)
- [Loading and merging rules](#loading-and-merging-rules)
- [Managing kubeconfig files with `kubectl config`](#managing-kubeconfig-files-with-kubectl-config)
- [Key points to remember](#key-points-to-remember)
- [Related discussion](#related-discussion)


## Example kubeconfig file

Let's walk through a few key details in the following example kubeconfig file to help you understand its contents and structure:

 - In this example, the `current-context` option is specified with value `federal-context`. When the `current-context` option is specified, the option is used by default for all of the client connections to any of the clusters defined in the file.
 - The `federal-context` context specifies `green-user` as the default user. Therefore, clients with correct certificates are logged in as the green-user because those credentials are specified in the file.
 - In contrast to the green-user, who authenticates with certificates, the blue-user is configured to use tokens.
 - This kubeconfig file corresponds to a Kubernetes api-server that was launched with the `kube-apiserver --token-auth-file=tokens.csv` option, where tokens.csv contains:

   ```
   blue-user,blue-user,1
   mister-red,mister-red,2
   ```

Note that because each user authenticates using different methods, this api-server was launched with other options in addition to `--token-auth-file`. It is important to understand the different options that you can run in order to implement your authentication scheme. You should run only the options for the authentication methods that align with your security requirements and policies.

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

As you can see, a kubeconfig file can contain more information than what’s necessary for a single session, including details for several clusters, users, and contexts.

## Loading and merging rules

Before you start creating kubeconfig files, its important to understand how they are loaded and merged. The kubeconfig files in your clusters get loaded and merged based on the following:

 * Environment variables
 * Commands that you run from the terminal window
 * Configuration information in each kubeconfig file

The following loading and merging rules are ordered by their priority:

 1. Retrieve the kubeconfig files from disk using the following hierarchy and merge rules:
   1. If the `CommandLineLocation` (the value of the `kubeconfig` command line option) is set, then use this file only and do not merge any other kubeconfig files. Only one instance of this flag is allowed in your kubeconfig files.
   1. If `EnvVarLocation` (the value of $KUBECONFIG) is available, then use it to define all of the files that will be merged.
   1. Merge the files using the following rules:
      * Empty filenames are ignored. Note: Files with non-deserializable content cause errors.
      * The first time that a particular value or map key is found, then use that value or map key and ignore any subsequent occurrences.
        For example, if a file sets `CurrentContext` for the first time, then the context in that file is preserved and any other files that also set `CurrentContext` are ignored. Another example is if two files specify a `red-user`, then only the values from the first file's red-user are used. Even non-conflicting entries from the second file's red-user are ignored and not merged into the final kubeconfig file.
      * If only a single kubeconfig file exists, then skip merging and use the file in `HomeDirectoryLocation` (~/.kube/config).
 1. Determine what context to use based on the following priorities:
   1. Command line argument (kubectl config): Use the value specified for `context`.
   1. Kubeconfig file: Use the value specified for `current-context`.
   1. Set no context if the values are undefined.
 1. Determine what user and cluster information to use based on the following priorities:
    Note: At this point, the context might be undefined. Also, this check runs twice: once for user and then again for cluster.
   1. Command line argument (kubectl config): Use the value specified for `user` and `cluster`.
   1. Kubeconfig file: If `context` is specified, then use the value in the nested `cluster`.
   1. Set no user and cluster if the values are undefined.
 1. Determine the details of the cluster based on the following priorities:
    Note: At this point, the cluster might be undefined. Also, only the values defined in the first found cluster are used and all subsequent values are ignored.
   1. Command line argument (kubectl config): Use the values specified for `server`, `api-version`, `certificate-authority`, and `insecure-skip-tls-verify`.
   1. Kubeconfig file: Use the values specified in `cluster`.
   1. Ensure that a value for `server` is defined, otherwise throw an error.
 1. Determine the details of the user based on the following priorities:
    Note: Only the values defined in the first found user are used and all subsequent values are ignored.
   1. Command line argument (kubectl config): Use the values specified for `client-certificate`, `client-key`, `username`, `password`, and `token`.
   1. Kubeconfig file: Use the values specified in `user`.
   1. Ensure only a single authentication method is defined, otherwise throw an error.
 1. For any required information that's missing, either use default values or prompt the user for authentication information.

## Managing kubeconfig files with `kubectl config`

Use the `kubectl config` *`subcommand`* commands to easily create, add, update, or remove details from your kubeconfig files. See the [kubectl_config](kubectl/kubectl_config.md) reference topic for details about all the commands.

### Creating kubeconfig files

You can run `kube-up.sh` to create kubeconfig files or you can manually create them.

If you create your cluster by running `kube-up.sh`, a kubeconfig file is created for you. See [Creating a Kubernetes Cluster](../getting-started-guides/README.md) for information about getting started with `kube-up.sh`.

To manually create a kubeconfig file, you can either create the file itself or you can use the `kubectl config` commands. For example, you can use the example kubeconfig file in the section above as a template to create your own kubeconfig file. If you manually create the file yourself, you must name it `config` and then save it to your `$HOME/.kube/` directory.

To use the `kubectl config` commands, you simply run the commands from your terminal window. See the following examples for information about what commands to run, including how you can re-create the example kubeconfig file.

### `kubectl config` examples

Use the following examples to help you familiarize yourself with the `kubectrl config` commands.

Tip: Revise the flags in the following examples so that they contain your cluster's api-server endpoint information and then run them to create your own kubeconfig files.

#### Create the example kubeconfig file

In the following example, we create a copy the example kubeconfig file that we use above:

 1. Run the following commands from your terminal window:

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

 1. Run `$ kubectl config view` to view your changes or open the actual kubeconfig file in the `$HOME/.kube/config` directory.

#### Updating your kubeconfig files

In the following example, we use the kubectl commands to add and then verify new user, cluster, and context information:

 1. To add the `myself` user, `local-server` cluster, and a default context, run the following commands from your terminal window:

        ```console
        $ kubectl config set-credentials myself --username=admin --password=secret
	    $ kubectl config set-cluster local-server --server=http://localhost:8080
	    $ kubectl config set-context default-context --cluster=local-server --user=myself
	    $ kubectl config use-context default-context
	    $ kubectl config set contexts.default-context.namespace the-right-prefix
	    ```

 1. To view the changes, run `$ kubectl config view`, for example:

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


 1. Optional: You can also verify the new configuration information by opening the actual kubeconfig file in the `$HOME/.kube/config` directory, for example:

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

## Key points to remember

A few important points to keep in mind when creating or configuring your kubeconfig file:

- Its important to understand the loading and merging rules, especially for managing multiple clusters and kubeconfig files.

- Before you configure your kubeconfig files to contain a convenient authentication method, take a good look and really understand how your api-server is being launched to ensure that you meet your security requirements and policies.

- Make sure that your api-server is launched so that at least one user's credentials are defined in it. For example, see our "green-user" in the example above. Review the [Authentication](../admin/authentication.md) topic to better understand how to set up user authentication.

## Related discussion

For in-depth design discussion and to determine if change is in the pipeline, you can review http://issue.k8s.io/1755.

#### Related information

- [Authentication](../admin/authentication.md)
- [kube-apiserver](../admin/kube-apiserver.md)
- [Sharing cluster access](sharing-clusters.md)
- [kubectl_config](kubectl/kubectl_config.md)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubeconfig-file.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
