# srvexpand

srvexpand is a tool to generate non-trivial but regular services
from a description free of most boilerplate.

Currently targets only v1beta3, which isn't yet fully implemented.

## Usage
```
$ srvexpand myservice.json
$ srvexpand myservice.yaml
```

## Schema
```
type HierarchicalController struct {
	// Optional: Defaults to one
	Replicas int `yaml:"replicas,omitempty" json:"replicas,omitempty"`
	// Spec defines the behavior of a pod.
	Spec v1beta3.PodSpec `json:"spec,omitempty" yaml:"spec,omitempty"`
}

type ControllerMap map[string]HierarchicalController

type HierarchicalService struct {
	// Optional: Creates a service if specified: servicePort:containerPort
	// TODO: Support multiple protocols
	PortSpec string `yaml:"portSpec,omitempty" json:"portSpec,omitempty"`
	// Map of replication controllers to create
	ControllerMap ControllerMap `json:"controllers,omitempty" yaml:"controllers,omitempty"`
}

type ServiceMap map[string]HierarchicalService
```

## Example
```
foo:
  portSpec: 80:8080
  controllers:
    canary:
      replicas: 2
      spec:
        containers:
          - name: web
            image: me/myappserver:canary
    stable:
      replicas: 10
      spec:
        containers:
          - name: web
            image: me/myappserver:stable
bar:
  portSpec: 3306:3306
  controllers:
    solo:
      replicas: 1
      spec:
        containers:
          - name: db
            image: mysql
        volumes:
          - name: dbdir
```
Output:
```
- kind: Service
  apiVersion: v1beta3
  metadata:
    name: foo
    creationTimestamp: "null"
    labels:
      service: foo
  spec:
    port: 80
    selector:
      service: foo
    containerPort: 8080
  status: {}
- kind: PodTemplate
  apiVersion: v1beta3
  metadata:
    name: foo-canary
    creationTimestamp: "null"
    labels:
      service: foo
      track: canary
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        service: foo
        track: canary
    spec:
      volumes: []
      containers:
      - name: web
        image: me/myappserver:canary
        imagePullPolicy: ""
      restartPolicy: {}
- kind: ReplicationController
  apiVersion: v1beta3
  metadata:
    name: foo-canary
    creationTimestamp: "null"
    labels:
      service: foo
      track: canary
  spec:
    replicas: 2
    selector:
      service: foo
      track: canary
    template:
      kind: PodTemplate
      name: foo-canary
      apiVersion: v1beta3
  status:
    replicas: 0
- kind: PodTemplate
  apiVersion: v1beta3
  metadata:
    name: foo-stable
    creationTimestamp: "null"
    labels:
      service: foo
      track: stable
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        service: foo
        track: stable
    spec:
      volumes: []
      containers:
      - name: web
        image: me/myappserver:stable
        imagePullPolicy: ""
      restartPolicy: {}
- kind: ReplicationController
  apiVersion: v1beta3
  metadata:
    name: foo-stable
    creationTimestamp: "null"
    labels:
      service: foo
      track: stable
  spec:
    replicas: 10
    selector:
      service: foo
      track: stable
    template:
      kind: PodTemplate
      name: foo-stable
      apiVersion: v1beta3
  status:
    replicas: 0
- kind: Service
  apiVersion: v1beta3
  metadata:
    name: bar
    creationTimestamp: "null"
    labels:
      service: bar
  spec:
    port: 3306
    selector:
      service: bar
    containerPort: 3306
  status: {}
- kind: PodTemplate
  apiVersion: v1beta3
  metadata:
    name: bar-solo
    creationTimestamp: "null"
    labels:
      service: bar
      track: solo
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        service: bar
        track: solo
    spec:
      volumes:
      - name: dbdir
        source: null
      containers:
      - name: db
        image: mysql
        imagePullPolicy: ""
      restartPolicy: {}
- kind: ReplicationController
  apiVersion: v1beta3
  metadata:
    name: bar-solo
    creationTimestamp: "null"
    labels:
      service: bar
      track: solo
  spec:
    replicas: 1
    selector:
      service: bar
      track: solo
    template:
      kind: PodTemplate
      name: bar-solo
      apiVersion: v1beta3
  status:
    replicas: 0

```