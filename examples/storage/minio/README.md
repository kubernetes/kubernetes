# Cloud Native Deployment of Minio using Kubernetes

## Table of Contents

  - [Prerequisites](#prerequisites)
  - [Minio Docker](#minio-docker)
  - [Get Started](#get-started)

  Minio is an AWS S3 compatible, object storage server built for cloud applications and devops.

  The following document describes the process to deploy [Minio](https://minio.io/) on Kubernetes. Minio is a _cloud native_ application, meaning
  Minio understands that it is running within a cluster manager, and uses this cluster management infrastructure for allocation of compute and storage
  resources.

  This example uses some of the core components of Kubernetes:

  - [_Pods_](https://kubernetes.io/docs/user-guide/pods/)
  - [_Services_](https://kubernetes.io/docs/user-guide/services/)
  - [_Deployments_](https://kubernetes.io/docs/user-guide/deployments/)
  - [_Persistent Volume Claims_](https://kubernetes.io/docs/user-guide/persistent-volumes/#persistentvolumeclaims)

  ## Prerequisites

  This example assumes that you have a Kubernetes version >=1.4 cluster installed and running,
  and that you have installed the [`kubectl`](../../../docs/user-guide/kubectl/kubectl.md)
  command line tool somewhere in your path.  Please see the
  [getting started guides](../../../docs/getting-started-guides/)
  for installation instructions for your platform.

  ## Minio Docker

  The pods use the [official Minio Docker image](https://hub.docker.com/r/minio/minio/~/dockerfile/) from Docker Hub.

  ## Get Started

  But before creating the deployment, you need to create a persistent volume claim (PVC) to request storage for the Minio instance. Kubernetes looks out for PVs matching the PVC request in the cluster and binds it to the PVC automatically. Create a PersistentVolumeClaim by downloading the file [minio-standalone-pvc.yaml] (minio-standalone-pvc.yaml?raw=true) and running

  ```sh
  kubectl create -f minio-standalone-pvc.yaml
  ```

  A deployment encapsulates replica sets and pods — so, if a pod goes down, replication controller makes sure another pod comes up automatically. This way you won’t need to bother about pod failures and will have a stable Minio service available. Create the Minio Deployment by downloading the file [minio-standalone-deployment.yaml] (minio-standalone-deployment.yaml?raw=true) and running

  ```sh
  kubectl create -f minio-standalone-deployment.yaml
  ```

  Now that you have a Minio deployment running, you may either want to access it internally (within the cluster) or expose it as a Service onto an external (outside of your cluster, maybe public internet) IP address, depending on your use case. You can achieve this using Services. There are 3 major service types — default type is ClusterIP, which exposes a service to connection from inside the cluster. NodePort and LoadBalancer are two types that expose services to external traffic.

  In this example, we expose the Minio Deployment by creating a LoadBalancer service. Download the file [minio-standalone-service.yaml] (minio-standalone-service.yaml?raw=true) and running

  ```sh
  kubectl create -f minio-standalone-service.yaml
  ```

  Once you are done, cleanup the cluster using
  ```sh
  kubectl delete deployment minio-deployment \
  &&  kubectl delete pvc minio-pv-claim \
  && kubectl delete svc minio-service
  ```
