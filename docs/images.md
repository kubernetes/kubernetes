# Images
Each container in a pod has its own image.  Currently, the only type of image supported is a [Docker Image](https://docs.docker.com/userguide/dockerimages/).

You create your Docker image and push it to a registry before referring to it in a kubernetes pod.

The `image` property of a container supports the same syntax as the `docker` command does, including private registries and tags.

## Updating Images

The default pull policy is `PullIfNotPresent` which causes the Kubelet to not
pull an image if it already exists. If you would like to always force a pull
you must set a pull image policy of `PullAlways` or specify a `:latest` tag on
your image.

## Using a Private Registry
Private registries may require keys to read images from them.
Credentials can be provided in several ways:
  - Using Google Container Registy
    - Per-cluster
    - automatically configured on GCE/GKE
    - all pods can read the project's private registry
  - Configuring Nodes to Authenticate to a Private Registry 
    - all pods can read any configured private registries
    - requires node configuration by cluster administrator
  - Pre-pulling Images
    - all pods can use any images cached on a node
    - requires root access to all nodes to setup
  - Specifying ImagePullKeys on a Pod
    - only pods which provide own keys can access the private registry
Each option is described in more detail below.
   

### Using Google Container Registry

Kubernetes has native support for the [Google Container
Registry (GCR)](https://cloud.google.com/tools/container-registry/), when running on Google Compute
Engine (GCE).  If you are running your cluster on GCE or Google Container Engine (GKE), simply
use the full image name (e.g. gcr.io/my_project/image:tag).

All pods in a cluster will have read access to images in this registry.

The kubelet kubelet will authenticate to GCR using the instance's
Google service account.  The service account on the instance
will have a `https://www.googleapis.com/auth/devstorage.read_only`,
so it can pull from the project's GCR, but not push.

### Configuring Nodes to Authenticate to a Private Registry 
Docker stores keys for private registries in a `.dockercfg` file.  Create a config file by running
`docker login <registry>.<domain>` and then copy the resulting `.dockercfg` file to the root user's
`$HOME` directory (e.g. `/root/.dockercfg`) on each node in the cluster.

You must ensure all nodes in the cluster have the same `.dockercfg`.  Otherwise, pods will run on
some nodes and fail to run on others.  For example, if you use node autoscaling, then each instance
template needs to include the `.dockercfg` or mount a drive that contains it.

All pods will have read access to images in any private registry with keys in the `.dockercfg`.

### Pre-pulling Images

Be default, the kubelet will try to pull each image from the specified registry.
However, if the `imagePullPolicy` property of the container is set to `IfNotPresent` or `Never`,
then a local image is used (preferentially or exclusively, respectively).

If you want to rely on pre-pulled images as a substitute for registry authentication,
you must ensure all nodes in the cluster have the same pre-pulled images.

This can be used to preload certain images for speed or as an alternative to authenticating to a private registry.

All pods will have read access to any pre-pulled images.

### Specifying ImagePullKeys on a Pod
Kubernetes supports specifying registry keys on a pod.

First, create a `.dockercfg`, such as running `docker login <registry.domain>`.
Then put the resulting `.dockercfg` file into a [secret resource](../docs/secret.md).  For example:
```
cat > dockercfg <<EOF
{ 
   "https://docker.io/thisisfake": { 
     "email": "bob@example.com", 
     "auth": "secret" 
   } 
}
EOF
$ cat dockercfg | base64
eyAKICAgImh0dHBzOi8vZG9ja2VyLmlvL3RoaXNpc2Zha2UiOiB7IAogICAgICJlbWFpbCI6ICJib2JAZXhhbXBsZS5jb20iLCAKICAgICAiYXV0aCI6ICJzZWNyZXQiIAogICB9Cn0K

cat > secret.json <<EOF
{
  "apiVersion": "v1",
  "kind": "Secret",
  "metadata" : {
    "name": "myregistrykey",
  },  
  "type": "kubernetes.io/dockercfg",
  "data": {
    ".dockercfg":
      "eyAKICAgImh0dHBzOi8vZG9ja2VyLmlvL3RoaXNpc2Zha2UiOiB7IAogICAgICJlbWFpbCI6ICJib2JAZXhhbXBsZS5jb20iLCAKICAgICAiYXV0aCI6ICJzZWNyZXQiIAogICB9Cn0K",
  }
}
EOF
This process only needs to be done one time (per namespace).

$ kubectl create -f secret.json
secrets/myregistrykey
```

Now, you can create pods which reference that secret by adding an `imagePullSecrets`
section to a pod definition.
```
apiVersion: v1
kind: Pod
metadata:
  name: foo
spec:
  containers:
    - name: foo
      image: registry.example.com/bar/fo
  imagePullSecrets:
    - name: myregistrykey
```
This needs to be done for each pod that is using a private registry.
However, setting of this field can be automated by setting the imagePullSecrets
in a [serviceAccount](../docs/service_accounts.md) resource.

Currently, all pods will potentially have read access to any images which were
pulled using imagePullSecrets.  That is, imagePullSecrets does *NOT* protect your
images from being seen by other users in the cluster.  Our intent
is to fix that.

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
     - no kubernetes configuration required
   - Or, when on GCE/GKE, use the project's Google Container Registry.
     - will work better with cluster autoscaling than manual node configuration
   - Or, on a cluster where changing the node configuration is inconvenient, use `imagePullSecrets`.
  1. Cluster with a proprietary images, a few of which require stricter access control
     - Move sensitive data into a "Secret" resource, instead of packaging it in an image.
     - DO NOT use imagePullSecrets for this use case yet.
  1. A multi-tenant cluster where each tenant needs own private registry
     - NOT supported yet.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/images.md?pixel)]()
