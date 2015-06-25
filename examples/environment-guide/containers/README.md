Building
--------
For each container, the build steps are the same. The examples below
are for the `show` container. Replace `show` with `backend` for the
backend container.

GCR
---
    docker build -t gcr.io/<project-name>/show .
    gcloud docker push gcr.io/<project-name>/show

Docker Hub
----------
    docker build -t <username>/show .
    docker push <username>/show

Change Pod Definitions
----------------------
Edit both `show-rc.yaml` and `backend-rc.yaml` and replace the
specified `image:` with the one that you built.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/environment-guide/containers/README.md?pixel)]()
