# Graph Component for Kubernetes WebUI

This is the Graph component for the Kubernetes UI. The graph component uses the [d3 Force Layout](https://github.com/mbostock/d3/wiki/Force-Layout) to expose the structure and organization of the cluster. It contains a legend that lets the user filter the types of objects displayed. Modifier keys let the user zoom the graph, and select or pin individual objects. Objects can also be inspected to display their available properties.

## Data Source
By default, the data displayed by the Graph tab is collected from the Kubernetes api server and the Docker daemons, and assembled into a single JSON document exposed on a REST endpoint by the cluster-insight container available [here](https://registry.hub.docker.com/u/kubernetes/cluster-insight/) on DockerHub. Installation and usage instructions for the cotainer are provided [here](https://github.com/google/cluster-insight) on GitHub.

The data are cached by the container and refreshed periodically to throttle the load on the cluster. The application can poll the container for the document continuously or on demand. When new contents are retrieved from the container, the application transforms them into the shape displayed on the canvas using a pluggable transform engine that loads transforms from the assets folder. The default transform is declarative; it interprets JSON documents loaded from the same location.

Canned data is also available for use without cluster-insight. It's selectable using the 'cloud' button located above the canvas. The canned data is served from a file in the assets folder.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/www/master/components/graph/README.md?pixel)]()
