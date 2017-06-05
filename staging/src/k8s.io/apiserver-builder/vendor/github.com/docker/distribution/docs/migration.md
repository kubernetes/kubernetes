<!--[metadata]>
+++
draft = true
+++
<![end-metadata]-->

# Migrating a 1.0 registry to 2.0

TODO: This needs to be revised in light of Olivier's work

A few thoughts here:

There was no "1.0". There was an implementation of the Registry API V1 but only a version 0.9 of the service was released.
The image formats are not compatible in any way. One must convert v1 images to v2 images using a docker client or other tool.
One can migrate images from one version to the other by pulling images from the old registry and pushing them to the v2 registry.

-----

The Docker Registry 2.0 is backward compatible with images created by the earlier specification. If you are migrating a private registry to version 2.0, you should use the following process:

1. Configure and test a 2.0 registry image in a sandbox environment.

2. Back up up your production image storage.

	Your production image storage should reside on a volume or storage backend.
	Make sure you have a backup of its contents.

3. Stop your existing registry service.

4. Restart your registry with your tested 2.0 image.
