# Images
Each container in a pod has its own image.  Currently, the only type of image supported is a [Docker Image](https://docs.docker.com/userguide/dockerimages/).

You create your Docker image and push it to a registry before referring to it in a kubernetes pod.

The `image` property of a container supports the same syntax as the `docker` command does, including private registries and tags.

## Using a Private Registry

### Google Container Registry
Kubernetes has native support for the [Google Container Registry](https://cloud.google.com/tools/container-registry/), when running on Google Compute Engine.  If you are running your cluster on Google Compute Engine or Google Container Engine, simply use the full image name (e.g. gcr.io/my_project/image:tag) and the kubelet will automatically authenticate and pull down your private image.

### Other Private Registries
Docker stores keys for private registries in a `.dockercfg` file.  Create a config file by running `docker login <registry>.<domain>` and then copying the resulting `.dockercfg` file to the kubelet working dir.
The kubelet working dir varies by cloud provider.  It is `/` on GCE and `/home/core` on CoreOS.  You can determine the working dir by running this command:
`sudo ls -ld /proc/$(pidof kubelet)/cwd` on a kNode.

All users of the cluster will have access to any private registry in the `.dockercfg`.

## Preloading Images

Be default, the kubelet will try to pull each image from the specified registry.
However, if the `imagePullPolicy` property of the container is set to `IfNotPresent` or `Never`,
then a local image is used (preferentially or exclusively, respectively).

This can be used to preload certain images for speed or as an alternative to authenticating to a private registry.

Pull Policy is per-container, but any user of the cluster will have access to all local images.

## Updating Images

The default pull policy is `PullIfNotPresent` which causes the Kubelet to not pull an image if it already exists. If you would like to always force a pull you must set a pull image policy of `PullAlways` or specify a `:latest` tag on your image.
