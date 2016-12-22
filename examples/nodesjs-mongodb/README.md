## Node.js and MongoDB on Kubernetes

The following document describes the deployment of a basic Node.js and MongoDB web stack on Kubernetes.  Currently this example does not use replica sets for MongoDB.

For more a in-depth explanation of this example, please [read this post.](https://medium.com/google-cloud-platform-developer-advocates/running-a-mean-stack-on-google-cloud-platform-with-kubernetes-149ca81c2b5d)

### Prerequisites

This example assumes that you have a basic understanding of Kubernetes conecepts (Pods, Services, Replication Controllers), a Kubernetes cluster up and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](../../docs/getting-started-guides/) for installation instructions for your platform.

Note: This example was tested on [Google Container Engine](https://cloud.google.com/container-engine/docs/). Some optional commands require the [Google Cloud SDK](https://cloud.google.com/sdk/).

### Creating the MongoDB Service

The first thing to do is create the MongoDB Service.  This service is used by the other Pods in the cluster to find and connect to the MongoDB instance.

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    name: mongo
  name: mongo
spec:
  ports:
    - port: 27017
      targetPort: 27017
  selector:
    name: mongo
```

[Download file](mongo-service.yaml)

This service looks for all pods with the "mongo" tag, and creates a Service on port 27017 that targets port 27017 on the MongoDB pods. Port 27017 is the standard MongoDB port.

To start the service, run:

```sh
kubectl create -f examples/nodesjs-mongodb/mongo-service.yaml
```

### Creating the MongoDB Controller

Next, create the MongoDB instance that runs the Database.  Databases also need persistent storage, which will be different for each platform.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: mongo
  name: mongo-controller
spec:
  replicas: 1
  template:
    metadata:
      labels:
        name: mongo
    spec:
      containers:
      - image: mongo
        name: mongo
        ports:
        - name: mongo
          containerPort: 27017
          hostPort: 27017
        volumeMounts:
            - name: mongo-persistent-storage
              mountPath: /data/db
      volumes:
        - name: mongo-persistent-storage
          gcePersistentDisk:
            pdName: mongo-disk
            fsType: ext4
```

[Download file](mongo-controller.yaml)

Looking at this file from the bottom up:

First, it creates a volume called "mongo-persistent-storage."

In the above example, it is using a "gcePersistentDisk" to back the storage. This is only applicable if you are running your Kubernetes cluster in Google Cloud Platform.

If you don't already have a [Google Persistent Disk](https://cloud.google.com/compute/docs/disks) created in the same zone as your cluster, create a new disk in the same Google Compute Engine / Container Engine zone as your cluster with this command:

```sh
gcloud compute disks create --size=200GB --zone=$ZONE mongo-disk
```

If you are using AWS, replace the "volumes" section with this (untested):

```yaml
      volumes:
        - name: mongo-persistent-storage
          awsElasticBlockStore:
            volumeID: aws://{region}/{volume ID}
            fsType: ext4
```

If you don't have a EBS volume in the same region as your cluster, create a new EBS volume in the same region with this command (untested):

```sh
ec2-create-volume --size 200 --region $REGION --availability-zone $ZONE
```

This command will return a volume ID to use.

For other storage options (iSCSI, NFS, OpenStack), please follow the documentation.

Now that the volume is created and usable by Kubernetes, the next step is to create the Pod.

Looking at the container section: It uses the official MongoDB container, names itself "mongo", opens up port 27017, and mounts the disk to "/data/db" (where the mongo container expects the data to be).

Now looking at the rest of the file, it is creating a Replication Controller with one replica, called mongo-controller. It is important to use a Replication Controller and not just a Pod, as a Replication Controller will restart the instance in case it crashes.

Create this controller with this command:

```sh
kubectl create -f examples/nodesjs-mongodb/mongo-controller.yaml
```

At this point, MongoDB is up and running.

Note: There is no password protection or auth running on the database by default. Please keep this in mind!

### Creating the Node.js Service

The next step is to create the Node.js service. This service is what will be the endpoint for the web site, and will load balance requests to the Node.js instances.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web
  labels:
    name: web
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 3000
      protocol: TCP
  selector:
    name: web
```

[Download file](web-service.yaml)

This service is called "web," and it uses a [LoadBalancer](../../docs/user-guide/services.md#type-loadbalancer) to distribute traffic on port 80 to port 3000 running on Pods with the "web" tag. Port 80 is the standard HTTP port, and port 3000 is the standard Node.js port.

On Google Container Engine, a [network load balancer](https://cloud.google.com/compute/docs/load-balancing/network/) and [firewall rule](https://cloud.google.com/compute/docs/networking#addingafirewall) to allow traffic are automatically created.

To start the service, run:

```sh
kubectl create -f examples/nodesjs-mongodb/web-service.yaml
```

If you are running on a platform that does not support LoadBalancer (i.e Bare Metal), you need to use a [NodePort](../../docs/user-guide/services.md#type-nodeport) with your own load balancer.

You may also need to open appropriate Firewall ports to allow traffic.

### Creating the Node.js Controller

The final step is deploying the Node.js container that will run the application code. This container can easily by replaced by any other web serving frontend, such as Rails, LAMP, Java, Go, etc.

The most important thing to keep in mind is how to access the MongoDB service.

If you were running MongoDB and Node.js on the same server, you would access MongoDB like so:

```javascript
MongoClient.connect('mongodb://localhost:27017/database-name', function(err, db) { console.log(db); });
```

With this Kubernetes setup, that line of code would become:

```javascript
MongoClient.connect('mongodb://mongo:27017/database-name', function(err, db) { console.log(db); });
```

The MongoDB Service previously created tells Kubernetes to configure the cluster so 'mongo' points to the MongoDB instance created earlier.

#### Custom Container

You should have your own container that runs your Node.js code hosted in a container registry.

See [this example](https://medium.com/google-cloud-platform-developer-advocates/running-a-mean-stack-on-google-cloud-platform-with-kubernetes-149ca81c2b5d#8edc) to see how to make your own Node.js container.

Once you have created your container, create the web controller.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: web
  name: web-controller
spec:
  replicas: 2
  selector:
    name: web
  template:
    metadata:
      labels:
        name: web
    spec:
      containers:
      - image: <YOUR-CONTAINER>
        name: web
        ports:
        - containerPort: 3000
          name: http-server
```

[Download file](web-controller.yaml)

Replace <YOUR-CONTAINER> with the url of your container.

This Controller will create two replicas of the Node.js container, and each Node.js container will have the tag "web" and expose port 3000. The Service LoadBalancer will forward port 80 traffic to port 3000 automatically, along with load balancing traffic between the two instances.

To start the Controller, run:

```sh
kubectl create -f examples/nodesjs-mongodb/web-controller.yaml
```

#### Demo Container

If you DON'T want to create a custom container, you can use the following YAML file:

Note: You cannot run both Controllers at the same time, as they both try to control the same Pods.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: web
  name: web-controller
spec:
  replicas: 2
  selector:
    name: web
  template:
    metadata:
      labels:
        name: web
    spec:
      containers:
      - image: node:0.10.40
        command: ['/bin/sh', '-c']
        args: ['cd /home && git clone https://github.com/ijason/NodeJS-Sample-App.git demo && cd demo/EmployeeDB/ && npm install && sed -i -- ''s/localhost/mongo/g'' app.js && node app.js']
        name: web
        ports:
        - containerPort: 3000
          name: http-server
```

[Download file](web-controller-demo.yaml)

This will use the default Node.js container, and will pull and execute code at run time. This is not recommended; typically, your code should be part of the container.

To start the Controller, run:

```sh
kubectl create -f examples/nodesjs-mongodb/web-controller-demo.yaml
```

### Testing it out

Now that all the components are running, visit the IP address of the load balancer to access the website.

With Google Cloud Platform, get the IP address of all load balancers with the following command:

```sh
gcloud compute forwarding-rules list
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nodesjs-mongodb/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
