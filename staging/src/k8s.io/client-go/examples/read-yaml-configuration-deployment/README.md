# Reads the Specifications of one Deployment and one Service from a YAML File and Create and/or Update Accordingly

This example program demonstrates a classical use case "a deployment exposed by a service" by doing the following:

1. Read a YAML file with one deployment and one related service (default: configuration.yaml)

2. Split it in k8s resources

3. Decode deployment from YAML to JSON

4. Unmarshall deployment from JSON to k8s resource

5. Try to create the deployment, and try to update it, if creation failed

6. Decode service from YAML to JSON

7. Unmarshall service from JSON to k8s resource

8. Try to create the service, and try to update it, if creation failed

9. Show exposed \<ip-address>:\<port number> (showing ip-address only, if using a minikube installation)

You can play with different specifications to see updates. Or delete resources to test how creating
and updating works together.

You can, as well, adopt the source code from this example to write programs that manage
other types of resources through the Kubernetes API.

## Running this example

Make sure you have a Kubernetes cluster and `kubectl` is configured:

    kubectl get nodes

Compile this example on your workstation:

```
cd read-yaml-configuration-deployment
go build -o ./app
```

Now, run this application on your workstation with your local kubeconfig file:

```
./app
```

> Usage of ./app:
>
>   -f string
>
>   (optional) absolute path to the YAML configuration file (default "configuration.yaml")
>
>   -kubeconfig string
>
>   (optional) absolute path to the kubeconfig file (default "$HOME/.kube/config")
>

You should see outputs like the following:

```
Create deployment "nginx-deployment"
Deployment "nginx-deployment" created

Create service "nginx"
Service "nginx" created

Please view: http://192.168.99.100:30001

```

or

```
Create deployment "nginx-deployment"
Info: deployments.apps "nginx-deployment" already exists

Update deployment "nginx-deployment"
Deployment "nginx-deployment" updated

Create service "nginx"
Info: Service "nginx" is invalid: spec.ports[0].nodePort: Invalid value: 30001: provided port is already allocated

Update service "nginx"
Service "nginx-deployment" updated

Please view: http://192.168.99.100:30001

```

## Cleanup

You can clean up the created deployment and service with:

    kubectl delete -f ./configuration.yaml

Accordingly, using other used YAML configuration and kubeconfig file, respectively.


## ToDos

- [ ] Overcome limitations for specification size of deployment (2048) and service (1024)
- [ ] Update to newest k8s version



