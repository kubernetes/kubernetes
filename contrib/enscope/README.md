# enscope

Typically a configuration is comprised of a set of objects (e.g., a simple service, replication controller, and template). Within that configuration, objects may refer to each other by object reference (as with replication controller to template, as of v1beta3) and/or by label selector (as with service and replication controller to pods generated from the pod template).

If one wants to create multiple instances of that configuration, such as for dev and prod deployments (aka horizontal composition) or to embed in composite macro-services (aka hierarchical composition), the names must be uniquified and the label selectors must be scoped to just one instance of the configuration, by adding deployment-specific labels and label selector requirements (e.g., env=prod, app==coolapp). 

Enscope is a standalone minimally schema-aware transformation pass for this purpose. It identifies all names, references, label sets, and label selectors that must be uniquified/scoped. An alternative would be to use a generic templating mechanism, such as [Mustache](http://mustache.github.io), but the scoping mechanism would need to be reimplemented in every templating language, and it would also make configurations more complex.

Currently targets only v1beta3, which isn't yet fully implemented.

## Usage
```
$ enscope specFilename configFilename
```

## Scope schema
```
type EnscopeSpec struct {
	NameSuffix string            `json:"nameSuffix,omitempty" yaml:"nameSuffix,omitempty"`
	Labels     map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}
```

## Example
The following name suffix and labels applied to the output from the [contrib/srvexpand example](../srvexpand/README.md):
```
nameSuffix: -coolapp-prod
labels:
  app: coolapp
  env: prod
```
Output:
```
- apiVersion: v1beta3
  kind: Service
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: foo
    name: foo-coolapp-prod
  spec:
    containerPort: 8080
    port: 80
    selector:
      app: coolapp
      env: prod
      service: foo
  status: {}
- apiVersion: v1beta3
  kind: PodTemplate
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: foo
      track: canary
    name: foo-canary-coolapp-prod
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        app: coolapp
        env: prod
        service: foo
        track: canary
    spec:
      containers:
      - image: me/coolappserver:canary
        imagePullPolicy: ""
        name: web
      restartPolicy: {}
      volumes: []
- apiVersion: v1beta3
  kind: ReplicationController
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: foo
      track: canary
    name: foo-canary-coolapp-prod
  spec:
    replicas: 2
    selector:
      app: coolapp
      env: prod
      service: foo
      track: canary
    template:
      apiVersion: v1beta3
      kind: PodTemplate
      name: foo-canary-coolapp-prod
  status:
    replicas: 0
- apiVersion: v1beta3
  kind: PodTemplate
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: foo
      track: stable
    name: foo-stable-coolapp-prod
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        app: coolapp
        env: prod
        service: foo
        track: stable
    spec:
      containers:
      - image: me/coolappserver:stable
        imagePullPolicy: ""
        name: web
      restartPolicy: {}
      volumes: []
- apiVersion: v1beta3
  kind: ReplicationController
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: foo
      track: stable
    name: foo-stable-coolapp-prod
  spec:
    replicas: 10
    selector:
      app: coolapp
      env: prod
      service: foo
      track: stable
    template:
      apiVersion: v1beta3
      kind: PodTemplate
      name: foo-stable-coolapp-prod
  status:
    replicas: 0
- apiVersion: v1beta3
  kind: Service
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: bar
    name: bar-coolapp-prod
  spec:
    containerPort: 3306
    port: 3306
    selector:
      app: coolapp
      env: prod
      service: bar
  status: {}
- apiVersion: v1beta3
  kind: PodTemplate
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: bar
      track: solo
    name: bar-solo-coolapp-prod
  spec:
    metadata:
      creationTimestamp: "null"
      labels:
        app: coolapp
        env: prod
        service: bar
        track: solo
    spec:
      containers:
      - image: mysql
        imagePullPolicy: ""
        name: db
      restartPolicy: {}
      volumes:
      - name: dbdir
        source: null
- apiVersion: v1beta3
  kind: ReplicationController
  metadata:
    creationTimestamp: "null"
    labels:
      app: coolapp
      env: prod
      service: bar
      track: solo
    name: bar-solo-coolapp-prod
  spec:
    replicas: 1
    selector:
      app: coolapp
      env: prod
      service: bar
      track: solo
    template:
      apiVersion: v1beta3
      kind: PodTemplate
      name: bar-solo-coolapp-prod
  status:
    replicas: 0
```