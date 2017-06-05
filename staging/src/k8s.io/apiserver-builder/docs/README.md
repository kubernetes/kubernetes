# Apiserver Builder

Apiserver builder is a collection of libraries and tools to
build Kubernetes native extensions into their own
Kubernetes apiservers.

## Building a simple apiserver

1. Copy the [example directory](https://github.com/pwittrock/sample-apiserver/tree/helloworld-2/vendor/k8s.io/apiserver-builder/example) and find all `ACTION REQUIRED` sections.  Follow instructions.
 - main.go: update import statements and uncomment starting the server
 - apis/doc.go: set your domain name
 - .../yourapigroup: change package name to match your api group
 - .../yourapiversion: change package name to match your version
 - .../yourapiversion/doc.go: update +conversion-gen with your api group go package
 - .../yourapiversion/types.go: update type names, fields, and comment tags


2. Generate code
  - `export REPO=github.com/orgname/reponame`
  - create the file `boilerplate.go.txt` containing the boilerplate headers for generated code
  - generate registration code
    - `go run vendor/k8s.io/apiserver-builder/cmd/genwiring/main.go -i $(REPO)/pkg/apis/...`
  - generate conversion code
    - `go go run vendor/k8s.io/kubernetes/cmd/libs/go2idl/conversion-gen/main.go -i $(REPO)/pkg/apis/...  --extra-peer-dirs="k8s.io/apimachinery/pkg/apis/meta/v1,k8s.io/apimachinery/pkg/conversion,k8s.io/apimachinery/pkg/runtime" -o ~/apiserver-helloworld/src/  -O zz_generated.conversion --go-header-file boilerplate.go.txt`
  - generate deepcopy code
    - `go run vendor/k8s.io/kubernetes/cmd/libs/go2idl/deepcopy-gen/main.go -i $(REPO)/pkg/apis/... -o ~/apiserver-helloworld/src/ -O zz_generated.deepcopy --go-header-file boilerplate.go.txt`
  - generate openapi code
    - `go run vendor/k8s.io/kubernetes/cmd/libs/go2idl/openapi-gen/main.go  -i "$(REPO)/pkg/apis/...,k8s.io/apimachinery/pkg/apis/meta/v1,k8s.io/apimachinery/pkg/api/resource/,k8s.io/apimachinery/pkg/version/,k8s.io/apimachinery/pkg/runtime/,k8s.io/apimachinery/pkg//util/intstr/" --output-package "$(REPO)/pkg/openapi" --go-header-file boilerplate.go.txt`

3. Build the apiserver main.go
  - go build main.go

## Running the apiserver with delegated auth against minikube

- start [minikube](https://github.com/kubernetes/minikube)
  - `minikube start`
- copy `~/.kube/config` to `~/.kube/auth_config`
  - `kubectl config use-context minikube`
  - `cp ~/.kube/config ~/.kube/auth_config`
- add a `~/.kube/config` entry for your apiserver, using the minikube user
  - `kubectl config set-cluster mycluster --server=https://localhost:9443 --certificate-authority=/var/run/kubernetes/apiserver.crt` // Use the cluster you created and the minikube user
  - `kubectl config set-context mycluster --user=minikube --cluster=mycluster`
  - `kubectl config use-context mycluster`
- make the directory `/var/run/kubernetes` if it doesn't exist
  - `sudo mkdir /var/run/kubernetes`
  - `sudo chown $(whoami) /var/run/kubernetes`
- run the server with ` ./main --authentication-kubeconfig ~/.kube/auth_config --authorization-kubeconfig ~/.kube/auth_config --client-ca-file /var/run/kubernetes/apiserver.crt  --requestheader-client-ca-file /var/run/kubernetes/apiserver.crt --requestheader-username-headers=X-Remote-User --requestheader-group-headers=X-Remote-Group --requestheader-extra-headers-prefix=X-Remote-Extra- --etcd-servers=http://localhost:2379 --secure-port=9443 --tls-ca-file  /var/run/kubernetes/apiserver.crt  --print-bearer-token`
  - This will have the server use minikube for delegated auth

## Generating docs

1. Vendor `vendor/github.com/kubernetes-incubator/reference-docs`
2. Start the server and look in the output for the curl command with a bearer token
3. `curl -k -H "Authorization: Bearer <bearer>" https://localhost:9443/swagger.json` > docs/openapi-spec/swagger.json
4. Build the static docs
  - `go run vendor/github.com/kubernetes-incubator/reference-docs/main.go --doc-type open-api --allow-errors --use-tags --config-dir docs --gen-open-api-dir vendor/github.com/kubernetes-incubator/reference-docs/gen_open_api`
  - `docker run -v $(shell pwd)/docs/includes:/source -v $(shell pwd)/docs/build:/build -v $(shell pwd)/docs/:/manifest pwittrock/brodocs`
5. Open docs file in browser `docs/build/index.html`

## Using apiserver-builder libraries directly (without generating code)