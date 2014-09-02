# Kubernetes Proposal - Images
## Problem/Rationale
Kubernetes creates Docker containers from images stored in Docker registries. It does not currently track and store any information about images; it merely pulls and stores them locally on a minion as part of the pod creation process.

Adding information related to images - image repositories, the images themselves, tags, and metadata - as resources in an image component will provide foundational support for several use cases, listed below.

### Use case: build downstream images when an upstream image repository changes
- Add ability to watch an image repository and be notified when changes occur
- Another component could automatically rebuild your image after an upstream change
- Being able to identify the important resource to watch without having to configure the remote system to notify you of changes is a valuable abstraction

**Example**: You create a new image based on SomeUser/AwesomeImage:latest. When the “latest” upstream tag is updated to point at a new image ID, you may want to be notified of the upstream change, and/or have your downstream image automatically rebuilt in response to the upstream update.

### Use case: modify image metadata without creating additional images
- Metadata for images = environment variables, exposed ports, memory & CPU restrictions, etc.
- Metadata provides value as an input to pod template generation; we therefore need to be able to store and access this metadata
- Add ability to combine an image and its default metadata with your overrides

**Example**: I find a great MySQL image that some else has created, but I want to tweak some of the values for the image’s environment variables. I don’t want to build my own copy of the Docker image with my settings applied because I want to subscribe to the “upstream” image and redeploy my containers when new images arrive.

**Questions**:

- How do we handle a lot of people making small tweaks to upstream images?
  - Does each person store the changes in their own IR?
- This does open you up to the situation where an image can run with one set of the environment variables but in turn fail with another set.  A situation that does not exist if the variables are bound at build time.

### Use case: consistent view of images over time
- Allows metadata about a particular version of an image that was used in a deployment to be captured for audit / historical review

### Use case: track image changes over time / “deployment contract”
- An image’s metadata can be seen as a deployment contract - what environment variables are used, what ports are exposed, etc.
- A deployment component could generate new a pod template when an image repository changes and compare it to the running pod to ensure nothing will break based on metadata changes

**Example**: You deploy a pod that uses an upstream Redis image. The image repository is updated, but the deployment component determines that the new pod template isn’t compatible with the current one, because the newer Redis image changed which port is exposed.

### Use case: Generate new pod template from an image’s metadata
- A configuration generation system can use metadata about the most recent version of an image to generate a pod template

### Use case: unified view of image repositories and images across multiple registries
- Provide a unified virtual view of disparate Docker registries, image repositories and images

**Example**: A PaaS operator pre-configures a set of image repositories from various Docker registries that can be shared by all of the PaaS’s users. Users go to one place to select images instead of searching on the Internet for a registry + image repository.

### Use case: track referenced and “in-use” images and remove unused images
- Images will accumulate over time, and many of them will no longer be referenced/used by any active or recent deployments
- Add ability to keep track of which images are possibly no longer relevant so they can be removed

### Use case: prevent users from using more than a reasonable amount of image layer storage on disk
- This is more of a registry use case, but depending on how things are implemented, it could be relevant to the image component
- This is presumably only talking about layers unique to the user’s image
- “Reasonable amount” should be large enough to work for most users
- The goal is to prevent bad actors from trying to fill up the system with huge garbage images (denial of service)

### Use case: be able to specify and control how many previous versions of an image that should be preserved
- For supporting rollbacks
- This may actually be a deployment & image tracking issue
  - Deployments support rollbacks
  - If the system can track which images haven’t been in use recently, it can purge old ones
- Or should part of this be implemented in a registry?
  - e.g. keep tag history per docker image repository
  - Limit # of tags per repo
  - Auto prune any image that was previously tagged but is now <= n (current) - j (however many old versions to keep)

### Use case: minimize impacts from images with security vulnerabilities
- Allow administrator to mark an image as having security issues
  - Should users be allowed to mark images they own?
- Prevent images with security issues from being deployed
- Allow users to see if they’re running anything based on images with security issues
- Possibly allow administrator to terminate all containers using image x (where x has a horrible security vulnerability)

### Use case: restrict access to image repositories only to those users who have access to the project to which the image repository belongs
- May require changes to the registry spec
- May not work with all registries


## Proposed Design
### ImageRepository
An ImageRepository is a type that records information about a collection of related images. It may reference a Docker image repository on a Docker registry, but this is optional. Its fields include:

- name, unique within a project or other scope
- reference to Docker registry image repository (optional)
- labels
- metadata to override (add to or modify) the Docker image’s embedded metadata
- metadata to clear (remove) from the Docker image’s embedded metadata
- tags

Add registry and storage for ImageRepository and register /imageRepositories with the apiserver.

### Image
An Image is an immutable type that records information about an image in a Docker image repository. Its fields include:

- labels
- metadata (the combination of the Docker image’s metadata along with the modifications from the ImageRepository)
- reference to image in a Docker registry/image repository

A new image must be created to make changes to an existing image (i.e. to change the metadata or labels).

Add registry and storage for Image and register /images with the apiserver.


## Docker registry - ImageRepository synchronization
### Option 1 - registry hook (preferred)
For registries that support executing hooks when an image/tag is pushed, a user configures the registry to invoke an image component webhook whenever a new image/tag is added to their Docker image repository.

When a user pushes an image/tag to a Docker registry, the registry posts a json payload to the image component and provides the image repository name, image ID, image metadata, and the new tag. Upon receiving the payload, the ImageRepository’s metadata overrides are applied to the image’s metadata and a new Image is created. The ImageRepository’s map of tags is updated as well.

**Notes**:

- the open-source docker-registry project does not currently have such a hook, but we have a fork with this feature added and hope that the upstream community will accept it.
- the Docker Hub’s webhook payload provides the image repository name, but it only supplies image IDs if new layers were pushed, and it never supplies image metadata or tag information. A specialized Docker Hub webhook handler is likely required, at least in the short term. When the image component's Hub webhook is invoked, it would pull the latest information from the Hub and then update the Image and ImageRepository information accordingly.

### Option 2 - polling of registry
For registries that don’t support image/tag push hooks, a DockerRegistryImageRepositoryWatcher can be configured to poll for changes to an image repository in a Docker registry. It would query the registry at a configurable interval, updating the list of tags and image metadata for any images not currently present in the image component.
