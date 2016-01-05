<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/porting-steps/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

App Porting Example
========
This example shows porting a very simple two-tier app from running
locally, to in containers, and finally on Kubernetes. Each
subdirectory contains a copy of the same app, and a simple README for
running in that environment. Feel free to run each version, or just
read on and examine the changes necessary.

Local
-----
The app is a simple guestbook, written in go. It uses mysql for a
database. It could just as easily be written in any language. The
version in the [`local/`](local/) directory is what you might write
when starting development, and running in a single IaaS instance. You
run mysql on your local machine, and build the app using development
tools installed on your machine. The app expects mysql to be running
on the same machine, as it connects to `localhost` to connect with the
database.

Containers
----------
This copy, in [`containers/`](containers/), gets most of the benefits
of containerization. The build of the code is happening inside a
Docker container, using the official golang image. This makes building
anywhere the same, and not dependant on the version of the dev tools
installed on your system. It always downloads the latest library
dependencies on build as well. The mysql container, again, is the same
everywhere it is run. You don't have to install and maintain mysql on
your machine.

The main change here is the addition of a [`Dockerfile`](containers/Dockerfile):

```
FROM golang:1.4-onbuild
EXPOSE 8080
```

The next major change, is how the app connects to the database:

```go
--- local/app.go	2015-05-14 10:28:44.938793915 -0700
+++ containers/app.go	2015-05-14 10:28:44.938793915 -0700
@@ -6,13 +6,16 @@
 	"html/template"
 	"log"
 	"net/http"
+	"os"
 	"time"
 
 	_ "github.com/go-sql-driver/mysql"
 )
 
 func connect() (*sql.DB, error) {
-	db, err := sql.Open("mysql", "root:secret@tcp(localhost:3306)/?parseTime=true")
+	dbpw := os.Getenv("DB_PW")
+	connect := fmt.Sprintf("root:%v@tcp(mysql-hostname:3306)/?parseTime=true", dbpw)
+	db, err := sql.Open("mysql", connect)
 	if err != nil {
 		return db, fmt.Errorf("Error opening db: %v", err)
 	}
```

Instead of using 'localhost', it uses 'mysql-hostname'. We use
Docker's [container
linking](https://docs.docker.com/userguide/dockerlinks/) to make the
database available as 'mysql-hostname' in the app container.

We also changed the app to allow the database password to be passed as
an environment variable. This made sense, as the database password is
set when running a new mysql container, and we wanted to send the
password to the app in the same way. We no longer have to manage the
password configuration of a mysql database installed on a system.

Here you are still running the two pieces on the same system, though
in their separate containers. You could easily run multiple app
containers on the same system, but would have to broker ports, and
figure out load balancing. If you wanted to run the app and database
on separate systems, you would have to figure out networking and
discovery, as the Docker linking feature does not work across
hosts. Enter Kubernetes:

Kubernetes
----------
Porting the app to run on Kubernetes will take care of the above
shortcomings. Given a cluster, Kubernetes will run the database and
front ends on any system, manage networking, and provide
discovery. The version of the app in [`k8s/`](k8s/) will run mysql and
two replicas of the front end on a Kubernetes cluster.

Since Kuberenetes is declarative. We need to add a few definition
files to our project, instead of keeping a playbook of docker
commands. We add:

```
mysql.yaml
twotier.yaml
```

These define the pods to run, and the services to make them
discoverable.  We keep the Dockerfile the same, but change the app
slightly to discover mysql in the Kubernetes environment:

```go
--- containers/app.go	2015-05-14 10:28:44.938793915 -0700
+++ k8s/app.go	2015-05-14 10:28:44.938793915 -0700
@@ -14,7 +14,9 @@
 
 func connect() (*sql.DB, error) {
 	dbpw := os.Getenv("DB_PW")
-	connect := fmt.Sprintf("root:%v@tcp(mysql-hostname:3306)/?parseTime=true", dbpw)
+	mysqlHost := os.Getenv("MYSQL_SERVICE_HOST")
+	mysqlPort := os.Getenv("MYSQL_SERVICE_PORT")
+	connect := fmt.Sprintf("root:%v@tcp(%v:%v)/?parseTime=true", dbpw, mysqlHost, mysqlPort)
 	db, err := sql.Open("mysql", connect)
 	if err != nil {
 		return db, fmt.Errorf("Error opening db: %v", err)
```

The app is coded to expect a service named 'mysql' to exist. In the
[`mysql.yaml`](k8s/mysql.yaml), under `kind: "Service"` we have `name:
"mysql"`. The app uses the Kubernetes provided environment variables
to locate the mysql service.

It is still expecting the password to be provided as an environment
variable. You can see this in [`mysql.yaml`](k8s/mysql.yaml) and
[`twotier.yaml`](k8s/twotier.yaml).

Bonus: Secret Store
-------------------
The app is already minimally ported to run in Kubernetes, but we can
take advantage of some more features. The version of this example in
the [`secret/`](secret/) directory makes use of Kubernetes
[secrets](https://github.com/docs/secrets.md). We are going to use the secrets feature
to store the password centrally of our mysql database. First we add a
file [`password.yaml`](secret/password.yaml) that defines the secret:

```yaml
apiVersion: "v1"
kind: "Secret"
metadata:
  name: "mysql-pw"
data:
  password: "bXlzZWNyZXRwYXNzd29yZA=="
```

Then we modify the pod definition of both the mysql and front end pods
to remove the password environment variables, and add the volume mount
definitions for the secret:

```yaml
--- k8s/mysql.yaml	2015-07-13 14:40:54.509756256 -0700
+++ secret/mysql.yaml	2015-07-13 14:32:16.259056145 -0700
@@ -15,12 +15,12 @@
           gcePersistentDisk:
             pdName: "mysql-disk"
             fsType: "ext4"
+        - name: "password"
+          secret:
+            secretName: "mysql-pw"
         containers:
         - name: "mysql"
-          image: "mysql:latest"
-          env:
-          - name: MYSQL_ROOT_PASSWORD
-            value: mysecretpassword
+          image: "gcr.io/google-samples/mysql:secret"
           ports:
           - name: "mysql"
             containerPort: 3306
@@ -28,6 +28,9 @@
           volumeMounts:
           - name: "mysql-vol"
             mountPath: "/var/lib/mysql"
+          - name: "password"
+            mountPath: "/etc/mysql-password"
+            readOnly: true
```

```yaml
--- k8s/twotier.yaml	2015-07-13 14:42:19.875519381 -0700
+++ secret/twotier.yaml	2015-07-13 14:32:17.711086045 -0700
@@ -10,17 +10,22 @@
         labels:
           role: "front"
       spec:
+        volumes:
+        - name: "password"
+          secret:
+            secretName: "mysql-pw"
         containers:
         - name: "twotier"
-          env:
-          - name: DB_PW
-            value: mysecretpassword
-          image: "gcr.io/google-samples/steps-twotier:k8s"
+          image: "gcr.io/google-samples/steps-twotier:secret"
           ports:
           - name: "http-server"
             hostPort: 80
             containerPort: 8080
             protocol: "TCP"
+          volumeMounts:
+          - name: "password"
+            mountPath: "/etc/mysql-password"
+            readOnly: true
```

We then modify our app to read the password from the mounted file, instead of the environment variable:

```go
--- k8s/app.go	2015-05-14 15:22:51.851319468 -0700
+++ secret/app.go	2015-05-15 14:32:31.111850594 -0700
@@ -4,6 +4,7 @@
 	"database/sql"
 	"fmt"
 	"html/template"
+	"io/ioutil"
 	"log"
 	"net/http"
 	"os"
@@ -13,10 +14,13 @@
 )
 
 func connect() (*sql.DB, error) {
-	dbpw := os.Getenv("DB_PW")
+	dbpw, err := ioutil.ReadFile("/etc/mysql-password/password")
+	if err != nil {
+		return nil, fmt.Errorf("Error reading db password: %v", err)
+	}
 	mysqlHost := os.Getenv("MYSQL_SERVICE_HOST")
 	mysqlPort := os.Getenv("MYSQL_SERVICE_PORT")
-	connect := fmt.Sprintf("root:%v@tcp(%v:%v)/?parseTime=true", dbpw, mysqlHost, mysqlPort)
+	connect := fmt.Sprintf("root:%v@tcp(%v:%v)/?parseTime=true", string(dbpw), mysqlHost, mysqlPort)
 	db, err := sql.Open("mysql", connect)
 	if err != nil {
 		return db, fmt.Errorf("Error opening db: %v", err)
```

For the mysql container, it is a bit trickier. We were originally
using the public mysql image, but we will need to tweak it to read the
password from a file. For this we add a new
[`mysql/Dockerfile`](secret/mysql/Dockerfile) that contains:

```
FROM mysql:latest
CMD export MYSQL_ROOT_PASSWORD=$(cat /etc/mysql-password/password); /entrypoint.sh mysqld
```

We are setting the password variable on the `CMD` line of the
Dockerfile, which gets evaluated at runtime. We then run the command
from the [original
Dockerfile](https://github.com/docker-library/mysql/blob/master/5.6/Dockerfile). Now
we use our customized mysql image, instead of the public image.





<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/porting-steps/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
