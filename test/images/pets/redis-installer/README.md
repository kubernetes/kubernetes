# Redis statefulset e2e tester

The image in this directory is the init container for contrib/pets/redis but for one difference, it bakes a specific version of redis into the base image so we get deterministic test results without having to depend on a redis download server. Discussing the tradeoffs to either approach (download the version at runtime, or maintain an image per version) are outside the scope of this document.

You can execute the image locally via:
```
docker run -it k8s.gcr.io/redis-install-3.2.0:e2e --cmd --install-into=/opt --work-dir=/work-dir
```
To share the installation with other containers mount the appropriate volumes as `--install-into` and `--work-dir`, where `install-into` is the directory to install redis into, and `work-dir` is the directory to install the user/admin supplied on-{start,change} hook scripts.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/pets/redis/README.md?pixel)]()
