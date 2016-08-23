# kubernetes-discovery
An initial implementation of a Kubernetes discovery service using JSON Web Signatures.

This prototype is expected to be run by Kubernetes itself for the time being,
and will hopefully be merged into the core API at a later time.

## Requirements

Generate a CA cert save it to: /tmp/secret/ca.pem to run the service or unit tests. (will not be required for unit tests for long) Similarly when run within kubernetes we expect a secret to be provided at this location as well. (see below)

## Build And Run From Source

```
$ make WHAT=cmd/kubediscovery
$ _output/local/bin/linux/amd64/kubediscovery
2016/08/23 19:17:28 Listening for requests on port 9898.

```

## Running in Docker

This image is published temporarily on Docker Hub as dgoodwin/kubediscovery

`docker run --rm -p 9898:9898 -v /tmp/secret/ca.pem:/tmp/secret/ca.pem --name kubediscovery dgoodwin/kubediscovery`

## Running in Kubernetes

A dummy certificate is included in ca-secret.yaml.

```
kubectl create -f ca-secret.yaml
kubectl create -f kubediscovery.yaml
```

## Testing the API

`curl "http://localhost:9898/cluster-info/v1/?token-id=TOKENID"`

You should see JSON containing a signed payload. For code to verify and decode that payload see handler_test.go.
