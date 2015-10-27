<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Images

Each container in a pod has its own image.  Currently, the only type of image supported is a [Docker Image](https://docs.docker.com/userguide/dockerimages/).

You create your Docker image and push it to a registry before referring to it in a Kubernetes pod.

The `image` property of a container supports the same syntax as the `docker` command does, including private registries and tags.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Images](#images)
  - [Updating Images](#updating-images)
  - [Using a Private Registry](#using-a-private-registry)
    - [Using Google Container Registry](#using-google-container-registry)
    - [Configuring Nodes to Authenticate to a Private Repository](#configuring-nodes-to-authenticate-to-a-private-repository)
    - [Pre-pulling Images](#pre-pulling-images)
    - [Specifying ImagePullSecrets on a Pod](#specifying-imagepullsecrets-on-a-pod)
    - [Use Cases](#use-cases)

<!-- END MUNGE: GENERATED_TOC -->

## Updating Images

The default pull policy is `IfNotPresent` which causes the Kubelet to not
pull an image if it already exists. If you would like to always force a pull
you must set a pull image policy of `Always` or specify a `:latest` tag on
your image.

## Using a Private Registry

Private registries may require keys to read images from them.
Credentials can be provided in several ways:
  - Using Google Container Registry
    - Per-cluster
    - automatically configured on Google Compute Engine or Google Container Engine
    - all pods can read the project's private registry
  - Configuring Nodes to Authenticate to a Private Registry
    - all pods can read any configured private registries
    - requires node configuration by cluster administrator
  - Pre-pulling Images
    - all pods can use any images cached on a node
    - requires root access to all nodes to setup
  - Specifying ImagePullSecrets on a Pod
    - only pods which provide own keys can access the private registry
Each option is described in more detail below.


### Using Google Container Registry

Kubernetes has native support for the [Google Container
Registry (GCR)](https://cloud.google.com/tools/container-registry/), when running on Google Compute
Engine (GCE).  If you are running your cluster on GCE or Google Container Engine (GKE), simply
use the full image name (e.g. gcr.io/my_project/image:tag).

All pods in a cluster will have read access to images in this registry.

The kubelet will authenticate to GCR using the instance's
Google service account.  The service account on the instance
will have a `https://www.googleapis.com/auth/devstorage.read_only`,
so it can pull from the project's GCR, but not push.

### Configuring Nodes to Authenticate to a Private Repository

**Note:** if you are running on Google Container Engine (GKE), there will already be a `.dockercfg` on each node
with credentials for Google Container Registry.  You cannot use this approach.

**Note:** this approach is suitable if you can control node configuration.  It
will not work reliably on GCE, and any other cloud provider that does automatic
node replacement.

Docker stores keys for private registries in the `$HOME/.dockercfg` file.  If you put this
in the `$HOME` of `root` on a kubelet, then docker will use it.

Here are the recommended steps to configuring your nodes to use a private registry.  In this
example, run these on your desktop/laptop:
   1. run `docker login [server]` for each set of credentials you want to use.
   1. view `$HOME/.dockercfg` in an editor to ensure it contains just the credentials you want to use.
   1. get a list of your nodes
      - for example: `nodes=$(kubectl get nodes -o template --template='{{range.items}}{{.metadata.name}} {{end}}')`
   1. copy your local `.dockercfg` to the home directory of root on each node.
      - for example: `for n in $nodes; do scp ~/.dockercfg root@$n:/root/.dockercfg; done`

Verify by creating a pod that uses a private image, e.g.:

```yaml
$ cat <<EOF > /tmp/private-image-test-1.yaml
apiVersion: v1
kind: Pod
metadata:
  name: private-image-test-1
spec:
  containers:
    - name: uses-private-image
      image: $PRIVATE_IMAGE_NAME
      imagePullPolicy: Always
      command: [ "echo", "SUCCESS" ]
EOF
$ kubectl create -f /tmp/private-image-test-1.yaml
pods/private-image-test-1
$
```

If everything is working, then, after a few moments, you should see:

```console
$ kubectl logs private-image-test-1
SUCCESS
```

If it failed, then you will see:

```console
$ kubectl describe pods/private-image-test-1 | grep "Failed"
  Fri, 26 Jun 2015 15:36:13 -0700	Fri, 26 Jun 2015 15:39:13 -0700	19	{kubelet node-i2hq}	spec.containers{uses-private-image}	failed		Failed to pull image "user/privaterepo:v1": Error: image user/privaterepo:v1 not found
```


You must ensure all nodes in the cluster have the same `.dockercfg`.  Otherwise, pods will run on
some nodes and fail to run on others.  For example, if you use node autoscaling, then each instance
template needs to include the `.dockercfg` or mount a drive that contains it.

All pods will have read access to images in any private registry once private
registry keys are added to the `.dockercfg`.

**This was tested with a private docker repository as of 26 June with Kubernetes version v0.19.3.
It should also work for a private registry such as quay.io, but that has not been tested.**

### Pre-pulling Images

**Note:** if you are running on Google Container Engine (GKE), there will already be a `.dockercfg` on each node
with credentials for Google Container Registry.  You cannot use this approach.

**Note:** this approach is suitable if you can control node configuration.  It
will not work reliably on GCE, and any other cloud provider that does automatic
node replacement.

Be default, the kubelet will try to pull each image from the specified registry.
However, if the `imagePullPolicy` property of the container is set to `IfNotPresent` or `Never`,
then a local image is used (preferentially or exclusively, respectively).

If you want to rely on pre-pulled images as a substitute for registry authentication,
you must ensure all nodes in the cluster have the same pre-pulled images.

This can be used to preload certain images for speed or as an alternative to authenticating to a private registry.

All pods will have read access to any pre-pulled images.

### Specifying ImagePullSecrets on a Pod

**Note:** This approach is currently the recommended approach for GKE, GCE, and any cloud-providers
where node creation is automated.

Kubernetes supports specifying registry keys on a pod.

First, create a `.dockercfg`, such as running `docker login <registry.domain>`.
Then put the resulting `.dockercfg` file into a [secret resource](secrets.md).  For example:

```console
$ docker login
Username: janedoe
Password: ●●●●●●●●●●●
Email: jdoe@example.com
WARNING: login credentials saved in /Users/jdoe/.dockercfg.
Login Succeeded

$ echo $(cat ~/.dockercfg)
{ "https://index.docker.io/v1/": { "auth": "ZmFrZXBhc3N3b3JkMTIK", "email": "jdoe@example.com" } }

$ cat ~/.dockercfg | base64
eyAiaHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEvIjogeyAiYXV0aCI6ICJabUZyWlhCaGMzTjNiM0prTVRJSyIsICJlbWFpbCI6ICJqZG9lQGV4YW1wbGUuY29tIiB9IH0K

$ cat > /tmp/image-pull-secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: myregistrykey
data:
  .dockercfg: eyAiaHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEvIjogeyAiYXV0aCI6ICJabUZyWlhCaGMzTjNiM0prTVRJSyIsICJlbWFpbCI6ICJqZG9lQGV4YW1wbGUuY29tIiB9IH0K
type: kubernetes.io/dockercfg
EOF

$ kubectl create -f /tmp/image-pull-secret.yaml
secrets/myregistrykey
$
```

If you get the error message `error: no objects passed to create`, it may mean the base64 encoded string is invalid.
If you get an error message like `Secret "myregistrykey" is invalid: data[.dockercfg]: invalid value ...` it means
the data was successfully un-base64 encoded, but could not be parsed as a dockercfg file.

This process only needs to be done one time (per namespace).

Now, you can create pods which reference that secret by adding an `imagePullSecrets`
section to a pod definition.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: foo
spec:
  containers:
    - name: foo
      image: janedoe/awesomeapp:v1
  imagePullSecrets:
    - name: myregistrykey
```

This needs to be done for each pod that is using a private registry.
However, setting of this field can be automated by setting the imagePullSecrets
in a [serviceAccount](service-accounts.md) resource.

Currently, all pods will potentially have read access to any images which were
pulled using imagePullSecrets.  That is, imagePullSecrets does *NOT* protect your
images from being seen by other users in the cluster.  Our intent
is to fix that.

You can use this in conjunction with a per-node `.dockerfile`.  The credentials
will be merged.  This approach will work on Google Container Engine (GKE).

### Use Cases

There are a number of solutions for configuring private registries.  Here are some
common use cases and suggested solutions.

 1. Cluster running only non-proprietary (e.g open-source) images.  No need to hide images.
   - Use public images on the Docker hub.
     - no configuration required
     - on GCE/GKE, a local mirror is automatically used for improved speed and availability
 1. Cluster running some proprietary images which should be hidden to those outside the company, but
   visible to all cluster users.
   - Use a hosted private [Docker registry](https://docs.docker.com/registry/)
     - may be hosted on the [Docker Hub](https://hub.docker.com/account/signup/), or elsewhere.
     - manually configure .dockercfg on each node as described above
   - Or, run an internal private registry behind your firewall with open read access.
     - no Kubernetes configuration required
   - Or, when on GCE/GKE, use the project's Google Container Registry.
     - will work better with cluster autoscaling than manual node configuration
   - Or, on a cluster where changing the node configuration is inconvenient, use `imagePullSecrets`.
  1. Cluster with a proprietary images, a few of which require stricter access control
     - Move sensitive data into a "Secret" resource, instead of packaging it in an image.
     - DO NOT use imagePullSecrets for this use case yet.
  1. A multi-tenant cluster where each tenant needs own private registry
     - NOT supported yet.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/images.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
