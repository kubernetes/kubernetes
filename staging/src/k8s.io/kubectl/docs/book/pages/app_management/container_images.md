{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Override or set the Name and Tag for Container Images
{% endpanel %}

# Container Images

## Motivation

It may be useful to define the tags or digests of container images which are used across many Workloads.

Container image tags and digests are used to refer to a specific version or instance of a container
image - e.g. for the `nginx` container image you might use the tag `1.15.9` or `1.14.2`.

- Update the container image name or tag for multiple Workloads at once
- Increase visibility of the versions of container images being used within
  the project
- Set the image tag from external sources - such as environment variables
- Copy or Fork an existing Project and change the Image Tag for a container
- Change the registry used for an image

See [Bases and Variations](../app_customization/bases_and_variants.md) for more details on Copying Projects.

{% panel style="info", title="Reference" %}
- [images](../reference/kustomize.md#images)
{% endpanel %}

## images

It is possible to set image tags for container images through
the `kustomization.yaml` using the `images` field.  When `images` are
specified, Apply will override the images whose image name matches `name` with a new
tag.


| Field     | Description                                                              | Example Field | Example Result |
|-----------|--------------------------------------------------------------------------|----------| --- |
| `name`    | Match images with this image name| `name: nginx`| |
| `newTag`  | Override the image **tag** or **digest** for images whose image name matches `name`    | `newTag: new` | `nginx:old` -> `nginx:new` |
| `newName` | Override the image **name** for images whose image name matches `name`   | `newImage: nginx-special` | `nginx:old` -> `nginx-special:old` |

{% method %}

**Example:** Use `images` in the `kustomization.yaml` to update the container
images in `deployment.yaml`

Apply will set the `nginx` image to have the tag `1.8.0` - e.g. `nginx:1.8.0` and
change the image name to `nginx-special`.
This will set the name and tag for *all* images matching the *name*.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
images:
  - name: nginx # match images with this name
    newTag: 1.8.0 # override the tag
    newName: nginx-special # override the name
resources:
- deployment.yaml
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
```

**Applied:** The Resource that is Applied to the cluster

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      # The image has been changed
      - image: nginx-special:1.8.0
        name: nginx
```
{% endmethod %}


## Setting a Name

{% method %}
The name for an image may be set by specifying `newName` and the name of the old container image.
{% sample lang="yaml" %}
```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
images:
  - name: mycontainerregistry/myimage
    newName: differentregistry/myimage
```
{% endmethod %}

## Setting a Tag

{% method %}
The tag for an image may be set by specifying `newTag` and the name of the container image.
{% sample lang="yaml" %}
```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
images:
  - name: mycontainerregistry/myimage
    newTag: v1
```
{% endmethod %}

## Setting a Digest

{% method %}
The digest for an image may be set by specifying `digest` and the name of the container image.
{% sample lang="yaml" %}
```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
images:
  - name: alpine
    digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
```
{% endmethod %}


## Setting a Tag from the latest commit SHA

{% method %}
A common CI/CD pattern is to tag container images with the git commit SHA of source code.  e.g. if
the image name is `foo` and an image was built for the source code at commit `1bb359ccce344ca5d263cd257958ea035c978fd3`
then the container image would be `foo:1bb359ccce344ca5d263cd257958ea035c978fd3`.

A simple way to push an image that was just built without manually updating the image tags is to
download the [kustomize standalone](https://github.com/kubernetes-sigs/kustomize/) tool and run
`kustomize edit set image` command to update the tags for you.

**Example:** Set the latest git commit SHA as the image tag for `foo` images.

{% sample lang="yaml" %}
```bash
kustomize edit set image foo:$(git log -n 1 --pretty=format:"%H")
kubectl apply -f .
```
{% endmethod %}

## Setting a Tag from an Environment Variable

{% method %}
It is also possible to set a Tag from an environment variable using the same technique for setting from a commit SHA.

**Example:** Set the tag for the `foo` image to the value in the environment variable `FOO_IMAGE_TAG`.

{% sample lang="yaml" %}
```bash
kustomize edit set image foo:$FOO_IMAGE_TAG
kubectl apply -f .
```
{% endmethod %}

{% panel style="info", title="Committing Image Tag Updates" %}
The `kustomization.yaml` changes *may* be committed back to git so that they
can be audited.  When committing the image tag updates that have already
been pushed by a CI/CD system, be careful not to trigger new builds +
deployments for these changes.
{% endpanel %}
