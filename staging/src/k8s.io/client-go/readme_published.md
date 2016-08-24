### Layout

Each top level folder (e.g., 1.4) contains a release of clients and their dependencies. In each release, the clientset is in the folder `kubernetes`; the discovery client and the dynamic client are in their dedicated folders; the `tools` folder contains useful packages built atop of the aforementioned clients.

### Releases

client-go has the same release cycle as the Kubernetes main repository. For example, in the 1.4 release cycle, the contents in `1.4/` folder are subjected to changes. The `1.4/` folder will be stable once the 1.4 release is final, and new changes will go into the `1.5/` folder.

### Contributing code
Please send pull requests against the client packages in the Kubernetes main repository, and run the `/staging/src/k8s.io/client-go/copy.sh` script to update the staging area in the main repository. Changes in the staging area will be published to this repository every day.
