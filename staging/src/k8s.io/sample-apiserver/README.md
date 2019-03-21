# sample-apiserver

Demonstration of how to use the k8s.io/apiserver library to build a functional API server.

**Note:** go-get or vendor this package as `k8s.io/sample-apiserver`.

## Purpose

You may use this code if you want to build an Extension API Server to use with API Aggregation, or to build a stand-alone Kubernetes-style API server.

However, consider two other options:
  * **CRDs**:  if you just want to add a resource to your kubernetes cluster, then consider using Custom Resource Definition a.k.a CRDs.  They require less coding and rebasing.  Read about the differences between Custom Resource Definitions vs Extension API Servers [here](https://kubernetes.io/docs/concepts/api-extension/custom-resources).
  * **Apiserver-builder**: If you want to build an Extension API server, consider using [apiserver-builder](https://github.com/kubernetes-incubator/apiserver-builder) instead of this repo.  The Apiserver-builder is a complete framework for generating the apiserver, client libraries, and the installation program.

If you do decide to use this repository, then the recommended pattern is to fork this repository, modify it to add your types, and then periodically rebase your changes on top of this repo, to pick up improvements and bug fixes to the apiserver.


## Compatibility

HEAD of this repo will match HEAD of k8s.io/apiserver, k8s.io/apimachinery, and k8s.io/client-go.

## Where does it come from?

`sample-apiserver` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/sample-apiserver.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Running it stand-alone

During development it is helpful to run sample-apiserver stand-alone, i.e. without
a Kubernetes API server for authn/authz and without aggregation. This is possible, but needs
a couple of flags, keys and certs as described below. You will still need some kubeconfig,
e.g. `~/.kube/config`, but the Kubernetes cluster is not used for authn/z. A minikube or
hack/local-up-cluster.sh cluster will work.

Instead of trusting the aggregator inside kube-apiserver, the described setup uses local
client certificate based X.509 authentication and authorization. This means that the client
certificate is trusted by a CA and the passed certificate contains the group membership
to the `system:masters` group. As we disable delegated authorization with `--authorization-skip-lookup`,
only this superuser group is authorized.

1. First we need a CA to later sign the client certificate:

``` shell
openssl req -nodes -new -x509 -keyout ca.key -out ca.crt
```

2. Then we create a client cert signed by this CA for the user `development` in the superuser group
   `system:masters`:

``` shell
openssl req -out client.csr -new -newkey rsa:4096 -nodes -keyout client.key -subj "/CN=development/O=system:masters"
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt
```

3. As curl requires client certificates in p12 format with password, do the conversion:

``` shell
openssl pkcs12 -export -in ./client.crt -inkey ./client.key -out client.p12 -passout pass:password
```

4. With these keys and certs in-place, we start the server:

``` shell
etcd &
sample-apiserver --secure-port 8443 --etcd-servers http://127.0.0.1:2379 --v=7 \
	--client-ca-file ca.crt \
	--kubeconfig ~/.kube/config \
	--authentication-kubeconfig ~/.kube/config \
	--authorization-kubeconfig ~/.kube/config
```

The first kubeconfig is used for the shared informers to access Kubernetes resources. The second kubeconfig passed to `--authentication-kubeconfig` is used to satisfy the delegated authenticator. The third kubeconfig passed to `--authorized-kubeconfig` is used to satisfy the delegated authorizer. Neither the authenticator, nor the authorizer will actually be used: due to `--client-ca-file`, our development X.509 certificate is accepted and authenticates us as `system:masters` member. `system:masters` is the superuser group
such that delegated authorization is skipped.

5. Use curl to access the server using the client certificate in p12 format for authentication:

``` shell
curl -fv -k --cert client.p12:password \
	https://localhost:8443/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders
```

   Note: Recent OSX versions broke client certs with curl. On Mac try `brew install httpie` and then:

``` shell
http --verify=no --cert client.crt --cert-key client.key \
	https://localhost:8443/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders
```
