<!--[metadata]>
+++
title = "Understanding the Registry"
description = "Explains what the Registry is, basic use cases and requirements"
keywords = ["registry, on-prem, images, tags, repository, distribution, use cases, requirements"]
[menu.main]
parent="smn_registry"
weight=2
+++
<![end-metadata]-->

# Understanding the Registry

A registry is a storage and content delivery system, holding named Docker images, available in different tagged versions.

  > Example: the image `distribution/registry`, with tags `2.0` and `2.1`.

Users interact with a registry by using docker push and pull commands.

  > Example: `docker pull registry-1.docker.io/distribution/registry:2.1`.

Storage itself is delegated to drivers. The default storage driver is the local posix filesystem, which is suitable for development or small deployments. Additional cloud-based storage drivers like S3, Microsoft Azure, OpenStack Swift and Aliyun OSS are also supported. People looking into using other storage backends may do so by writing their own driver implementing the [Storage API](storagedrivers.md).

Since securing access to your hosted images is paramount, the Registry natively supports TLS and basic authentication.

The Registry GitHub repository includes additional information about advanced authentication and authorization methods. Only very large or public deployments are expected to extend the Registry in this way.

Finally, the Registry ships with a robust [notification system](notifications.md), calling webhooks in response to activity, and both extensive logging and reporting, mostly useful for large installations that want to collect metrics.

## Understanding image naming

Image names as used in typical docker commands reflect their origin:

 * `docker pull ubuntu` instructs docker to pull an image named `ubuntu` from the official Docker Hub. This is simply a shortcut for the longer `docker pull docker.io/library/ubuntu` command
 * `docker pull myregistrydomain:port/foo/bar` instructs docker to contact the registry located at `myregistrydomain:port` to find the image `foo/bar`

You can find out more about the various Docker commands dealing with images in the [official Docker engine documentation](https://docs.docker.com/reference/commandline/cli/).

## Use cases

Running your own Registry is a great solution to integrate with and complement your CI/CD system. In a typical workflow, a commit to your source revision control system would trigger a build on your CI system, which would then push a new image to your Registry if the build is successful. A notification from the Registry would then trigger a deployment on a staging environment, or notify other systems that a new image is available.

It's also an essential component if you want to quickly deploy a new image over a large cluster of machines.

Finally, it's the best way to distribute images inside an isolated network.

## Requirements

You absolutely need to be familiar with Docker, specifically with regard to pushing and pulling images. You must understand the difference between the daemon and the cli, and at least grasp basic concepts about networking.

Also, while just starting a registry is fairly easy, operating it in a production environment requires operational skills, just like any other service. You are expected to be familiar with systems availability and scalability, logging and log processing, systems monitoring, and security 101. Strong understanding of http and overall network communications, plus familiarity with golang are certainly useful as well for advanced operations or hacking.

## Next

Dive into [deploying your registry](deploying.md)
