# Cloud Native Deployment of Minio using Kubernetes

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Minio Standalone Server Deployment](#minio-standalone-server-deployment)
  - [Standalone Quickstart](#standalone-quickstart)
  - [Step 1: Create Persistent Volume Claim](#step-1-create-persistent-volume-claim)
  - [Step 2: Create Deployment](#step-2-create-minio-deployment)
  - [Step 3: Create LoadBalancer Service](#step-3-create-minio-service)
  - [Step 4: Resource cleanup](#step-4-resource-cleanup)
- [Minio Distributed Server Deployment](#minio-distributed-server-deployment)
  - [Distributed Quickstart](#distributed-quickstart)
  - [Step 1: Create Minio Headless Service](#step-1-create-minio-headless-service)
  - [Step 2: Create Minio Statefulset](#step-2-create-minio-statefulset)
  - [Step 3: Create LoadBalancer Service](#step-3-create-minio-service)
  - [Step 4: Resource cleanup](#step-4-resource-cleanup)

## Introduction
Minio is an AWS S3 compatible, object storage server built for cloud applications and devops. Minio is _cloud native_, meaning Minio understands that it is running within a cluster manager, and uses the cluster management infrastructure for allocation of compute and storage resources.

## Prerequisites

This example assumes that you have a Kubernetes version >=1.4 cluster installed and running, and that you have installed the [`kubectl`](https://kubernetes.io/docs/user-guide/prereqs/) command line tool in your path. Please see the
[getting started guides](https://kubernetes.io/docs/getting-started-guides/) for installation instructions for your platform.

## Minio Standalone Server Deployment

The following section describes the process to deploy standalone [Minio](https://minio.io/) server on Kubernetes. The deployment uses the [official Minio Docker image](https://hub.docker.com/r/minio/minio/~/dockerfile/) from Docker Hub.

This section uses following core components of Kubernetes:

- [_Pods_](https://kubernetes.io/docs/user-guide/pods/)
- [_Services_](https://kubernetes.io/docs/user-guide/services/)
- [_Deployments_](https://kubernetes.io/docs/user-guide/deployments/)
- [_Persistent Volume Claims_](https://kubernetes.io/docs/user-guide/persistent-volumes/#persistentvolumeclaims)

### Standalone Quickstart

Run the below commands to get started quickly

```sh
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-pvc.yaml?raw=true
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-deployment.yaml?raw=true
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-service.yaml?raw=true
```

### Step 1: Create Persistent Volume Claim

Minio needs persistent storage to store objects. If there is no
persistent storage, the data stored in Minio instance will be stored in the container file system and will be wiped off as soon as the container restarts.

Create a persistent volume claim (PVC) to request storage for the Minio instance. Kubernetes looks out for PVs matching the PVC request in the cluster and binds it to the PVC automatically.

This is the PVC description.

```sh
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  # This name uniquely identifies the PVC. Will be used in deployment below.
  name: minio-pv-claim
  annotations:
    volume.alpha.kubernetes.io/storage-class: anything
  labels:
    app: minio-storage-claim
spec:
  # Read more about access modes here: http://kubernetes.io/docs/user-guide/persistent-volumes/#access-modes
  accessModes:
    - ReadWriteOnce
  resources:
    # This is the request for storage. Should be available in the cluster.
    requests:
      storage: 10Gi
```

Create the PersistentVolumeClaim

```sh
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-pvc.yaml?raw=true
persistentvolumeclaim "minio-pv-claim" created
```

### Step 2: Create Minio Deployment

A deployment encapsulates replica sets and pods — so, if a pod goes down, replication controller makes sure another pod comes up automatically. This way you won’t need to bother about pod failures and will have a stable Minio service available.

This is the deployment description.

```sh
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  # This name uniquely identifies the Deployment
  name: minio-deployment
spec:
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        # Label is used as selector in the service.
        app: minio
    spec:
      # Refer to the PVC created earlier
      volumes:
      - name: storage
        persistentVolumeClaim:
          # Name of the PVC created earlier
          claimName: minio-pv-claim
      containers:
      - name: minio
        # Pulls the default Minio image from Docker Hub
        image: minio/minio
        command:
        - minio
        args:
        - server
        - /storage
        env:
        # Minio access key and secret key
        - name: MINIO_ACCESS_KEY
          value: "minio"
        - name: MINIO_SECRET_KEY
          value: "minio123"
        ports:
        - containerPort: 9000
          hostPort: 9000
        # Mount the volume into the pod
        volumeMounts:
        - name: storage # must match the volume name, above
          mountPath: "/storage"
```

Create the Deployment

```sh
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-deployment.yaml?raw=true
deployment "minio-deployment" created
```

### Step 3: Create Minio Service

Now that you have a Minio deployment running, you may either want to access it internally (within the cluster) or expose it as a Service onto an external (outside of your cluster, maybe public internet) IP address, depending on your use case. You can achieve this using Services. There are 3 major service types — default type is ClusterIP, which exposes a service to connection from inside the cluster. NodePort and LoadBalancer are two types that expose services to external traffic.

In this example, we expose the Minio Deployment by creating a LoadBalancer service. This is the service description.

```sh
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  type: LoadBalancer
  ports:
    - port: 9000
      targetPort: 9000
      protocol: TCP
  selector:
    app: minio
```
Create the Minio service

```sh
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-standalone-service.yaml?raw=true
service "minio-service" created
```

The `LoadBalancer` service takes couple of minutes to launch. To check if the service was created successfully, run the command

```sh
kubectl get svc minio-service
NAME            CLUSTER-IP     EXTERNAL-IP       PORT(S)          AGE
minio-service   10.55.248.23   104.199.249.165   9000:31852/TCP   1m
```

### Step 4: Resource cleanup

Once you are done, cleanup the cluster using
```sh
kubectl delete deployment minio-deployment \
&&  kubectl delete pvc minio-pv-claim \
&& kubectl delete svc minio-service
```

## Minio Distributed Server Deployment

The following document describes the process to deploy [distributed Minio](https://docs.minio.io/docs/distributed-minio-quickstart-guide) server on Kubernetes. This example uses the [official Minio Docker image](https://hub.docker.com/r/minio/minio/~/dockerfile/) from Docker Hub.

This example uses following core components of Kubernetes:

- [_Pods_](https://kubernetes.io/docs/user-guide/pods/)
- [_Services_](https://kubernetes.io/docs/user-guide/services/)
- [_Statefulsets_](https://kubernetes.io/docs/tutorials/stateful-application/basic-stateful-set/)

### Distributed Quickstart

Run the below commands to get started quickly

```sh
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-headless-service.yaml?raw=true
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-statefulset.yaml?raw=true
kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-service.yaml?raw=true
```

### Step 1: Create Minio Headless Service

Headless Service controls the domain within which StatefulSets are created. The domain managed by this Service takes the form: `$(service name).$(namespace).svc.cluster.local` (where “cluster.local” is the cluster domain), and the pods in this domain take the form: `$(pod-name-{i}).$(service name).$(namespace).svc.cluster.local`. This is required to get a DNS resolvable URL for each of the pods created within the Statefulset.

This is the Headless service description.

```sh
apiVersion: v1
kind: Service
metadata:
  name: minio
  labels:
    app: minio
spec:
  clusterIP: None
  ports:
    - port: 9000
      name: minio
  selector:
    app: minio
```

Create the Headless Service

```sh
$ kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-headless-service.yaml?raw=true
service "minio" created
```

### Step 2: Create Minio Statefulset

A StatefulSet provides a deterministic name and a unique identity to each pod, making it easy to deploy stateful distributed applications. To launch distributed Minio you need to pass drive locations as parameters to the minio server command. Then, you’ll need to run the same command on all the participating pods. StatefulSets offer a perfect way to handle this requirement.

This is the Statefulset description.

```sh
apiVersion: apps/v1beta1
kind: StatefulSet
metadata:
  name: minio
spec:
  serviceName: minio
  replicas: 4
  template:
    metadata:
      annotations:
        pod.alpha.kubernetes.io/initialized: "true"
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        env:
        - name: MINIO_ACCESS_KEY
          value: "minio"
        - name: MINIO_SECRET_KEY
          value: "minio123"
        image: minio/minio
        command:
        - minio
        args:
        - server
        - http://minio-0.minio.default.svc.cluster.local/data
        - http://minio-1.minio.default.svc.cluster.local/data
        - http://minio-2.minio.default.svc.cluster.local/data
        - http://minio-3.minio.default.svc.cluster.local/data
        ports:
        - containerPort: 9000
          hostPort: 9000
        # These volume mounts are persistent. Each pod in the PetSet
        # gets a volume mounted based on this field.
        volumeMounts:
        - name: data
          mountPath: /data
  # These are converted to volume claims by the controller
  # and mounted at the paths mentioned above.
  volumeClaimTemplates:
  - metadata:
      name: data
      annotations:
        volume.alpha.kubernetes.io/storage-class: anything
    spec:
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 10Gi
```

Create the Statefulset

```sh
$ kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-statefulset.yaml?raw=true
statefulset "minio" created
```

### Step 3: Create Minio Service

Now that you have a Minio statefulset running, you may either want to access it internally (within the cluster) or expose it as a Service onto an external (outside of your cluster, maybe public internet) IP address, depending on your use case. You can achieve this using Services. There are 3 major service types — default type is ClusterIP, which exposes a service to connection from inside the cluster. NodePort and LoadBalancer are two types that expose services to external traffic.

In this example, we expose the Minio Deployment by creating a LoadBalancer service. This is the service description.

```sh
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  type: LoadBalancer
  ports:
    - port: 9000
      targetPort: 9000
      protocol: TCP
  selector:
    app: minio
```
Create the Minio service

```sh
$ kubectl create -f https://github.com/kubernetes/kubernetes/blob/master/examples/storage/minio/minio-distributed-service.yaml?raw=true
service "minio-service" created
```

The `LoadBalancer` service takes couple of minutes to launch. To check if the service was created successfully, run the command

```sh
$ kubectl get svc minio-service
NAME            CLUSTER-IP     EXTERNAL-IP       PORT(S)          AGE
minio-service   10.55.248.23   104.199.249.165   9000:31852/TCP   1m
```

### Step 4: Resource cleanup

You can cleanup the cluster using
```sh
kubectl delete statefulset minio \
&&  kubectl delete svc minio \
&& kubectl delete svc minio-service
```
