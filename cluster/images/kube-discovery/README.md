### kube-discovery

An initial implementation of a Kubernetes discovery service using JSON Web Signatures.

This prototype is configured by kubeadm and run within Kubernetes itself.

## Requirements

This pod expects the cluster CA, endpoints list, and token map to exist in /tmp/secret. This allows us to pass them in as kubernetes secrets when deployed as a pod.

```
$ cd /tmp/secret
$ ls
ca.pem  endpoint-list.json  token-map.json
$ cat endpoint-list.json
["http://192.168.1.5:8080", "http://192.168.1.6:8080"]
$ cat token-map.json
{
    "TOKENID": "ABCDEF1234123456"
}
```

## Build And Run From Source

```
$ build/run.sh /bin/bash -c "KUBE_BUILD_PLATFORMS=linux/amd64 make WHAT=cmd/kube-discovery"
$ _output/dockerized/bin/linux/amd64/kube-discovery
2016/08/23 19:17:28 Listening for requests on port 9898.

```

## Running in Docker

This image is published at: gcr.io/google_containers/kube-discovery

`docker run -d -p 9898:9898 -v /tmp/secret/ca.pem:/tmp/secret/ca.pem -v /tmp/secret/endpoint-list.json:/tmp/secret/endpoint-list.json -v /tmp/secret/token-map.json:/tmp/secret/token-map.json --name kubediscovery gcr.io/google_containers/kube-discovery`

## Testing the API

`curl "http://localhost:9898/cluster-info/v1/?token-id=TOKENID"`

You should see JSON containing a signed payload. For code to verify and decode that payload see handler_test.go.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/kube-discovery/README.md?pixel)]()
