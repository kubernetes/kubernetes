
An assortment of compact kubectl examples

See also: [Kubectl overview](kubectl-overview.md) and [JsonPath guide](jsonpath.md).

## Creating Objects

```console
$ kubectl create -f ./file.yml                   # create resource(s) in a json or yaml file

$ kubectl create -f ./file1.yml -f ./file2.yaml  # create resource(s) in a json or yaml file

$ kubectl create -f ./dir                        # create resources in all .json, .yml, and .yaml files in dir

# Create from a URL
$ kubectl create -f http://www.fpaste.org/279276/48569091/raw/

# Create multiple YAML objects from stdin
$ cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - sleep
    - "1000000"
---
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep-less
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - sleep
    - "1000"
EOF

# Create a secret with several keys
$ cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: $(echo "s33msi4" | base64)
  username: $(echo "jane" | base64)
EOF

# TODO: kubectl-explain example
```


## Viewing, Finding Resources

```console
# Columnar output
$ kubectl get services                          # List all services in the namespace
$ kubectl get pods --all-namespaces             # List all pods in all namespaces
$ kubectl get pods -o wide                      # List all pods in the namespace, with more details
$ kubectl get rc <rc-name>                      # List a particular replication controller
$ kubectl get replicationcontroller <rc-name>   # List a particular RC

# Verbose output
$ kubectl describe nodes <node-name>
$ kubectl describe pods <pod-name>
$ kubectl describe pods/<pod-name>              # Equivalent to previous
$ kubectl describe pods <rc-name>               # Lists pods created by <rc-name> using common prefix

# List Services Sorted by Name
$ kubectl get services --sort-by=.metadata.name

# List pods Sorted by Restart Count
$ kubectl get pods --sort-by=.status.containerStatuses[0].restartCount

# Get the version label of all pods with label app=cassandra
$ kubectl get pods --selector=app=cassandra rc -o 'jsonpath={.items[*].metadata.labels.version}'

# Get ExternalIPs of all nodes
$ kubectl get nodes -o jsonpath='{.items[*].status.addresses[?(@.type=ExternalIP)].address}'

# List Names of Pods that belong to Particular RC
# "jq" command useful for transformations that are too complex for jsonpath
$ sel=$(./kubectl get rc <rc-name> --output=json | jq -j '.spec.selector | to_entries | .[] | "\(.key)=\(.value),"')
$ sel=${sel%?} # Remove trailing comma
$ pods=$(kubectl get pods --selector=$sel --output=jsonpath={.items..metadata.name})`

# Check which nodes are ready
$ kubectl get nodes -o jsonpath='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}'| tr ';' "\n"  | grep "Ready=True" 
```

## Modifying and Deleting Resources

```console
$ kubectl label pods <pod-name> new-label=awesome                  # Add a Label
$ kubectl annotate pods <pod-name> icon-url=http://goo.gl/XXBTWq   # Add an annotation

# TODO: examples of kubectl edit, patch, delete, replace, scale, and rolling-update commands.
```

## Interacting with running Pods

```console
$ kubectl logs <pod-name>         # dump pod logs (stdout)
$ kubectl logs -f <pod-name>      # stream pod logs (stdout) until canceled (ctrl-c) or timeout

$ kubectl run -i --tty busybox --image=busybox -- sh      # Run pod as interactive shell
$ kubectl attach <podname> -i                             # Attach to Running Container
$ kubectl port-forward <podname> <local-and-remote-port>  # Forward port of Pod to your local machine
$ kubectl port-forward <servicename> <port>               # Forward port to service
$ kubectl exec <pod-name> -- ls /                         # Run command in existing pod (1 container case) 
$ kubectl exec <pod-name> -c <container-name> -- ls /     # Run command in existing pod (multi-container case) 
```



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl-cheatsheet.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
