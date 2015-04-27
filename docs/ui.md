# Kubernetes UI instructions

## Kubernetes User Interface
Kubernetes currently supports a simple web user interface.

### Running locally
Start the server:

```sh
cluster/kubectl.sh proxy --www=$PWD/www
```

The UI should now be running on [localhost](http://localhost:8001/static/index.html#/groups//selector)

### Running remotely
When Kubernetes is deployed, the server deploys the UI, you can visit ```/static/index.html#/groups//selector``` on your master server.


### Interacting with the user interface.

The Kubernetes user interface is a query-based visualization of the Kubernetes API.  The user interface is defined by two functional primitives:

#### GroupBy
_GroupBy_ takes a label ```key``` as a parameter, places all objects with the same value for that key within a single group. For example ```/groups/host/selector``` groups pods by host.  ```/groups/name/selector``` groups pods by name.  Groups are hierarchical, for example ```/groups/name/host/selector``` first groups by pod name, and then by host.

#### Select
Select takes a [label selector](./labels.md) and uses it to filter, so only resources which match that label selector are displayed.  For example, ```/groups/host/selector/name=frontend```, shows pods, grouped by host, which have a label with the name `frontend`.


## Rebuilding the UI
The UI relies on [go-bindata](https://github.com/jteeuwen/go-bindata)

To install go-bindata:

```sh
go get github.com/jteeuwen/go-bindata/...
```

To rebuild the UI, run the following:

```sh
hack/build-ui.sh
```
