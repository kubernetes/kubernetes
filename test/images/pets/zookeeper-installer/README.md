# Zookeeper statefulset e2e tester

The image in this directory is the init container for contrib/pets/zookeeper but for one difference, it bakes a specific version of zookeeper into the base image so we get deterministic test results without having to depend on a zookeeper download server. Discussing the tradeoffs to either approach (download the version at runtime, or maintain an image per version) are outside the scope of this document.

You can execute the image locally via:
```
docker run -it registry.k8s.io/zookeeper-install-3.5.0-alpha:e2e --cmd --install-into=/opt --work-dir=/work-dir
```
To share the installation with other containers mount the appropriate volumes as `--install-into` and `--work-dir`, where `install-into` is the directory to install zookeeper into, and `work-dir` is the directory to install the user/admin supplied on-{start,change} hook scripts.

