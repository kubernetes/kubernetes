kubernetes_nginx
----------------

This Dockerfile will build a docker image with nginx running (~130MB). It can be built with the following:

    docker build -t kubernetes_nginx:latest .

The running container makes some assumptions about items available to it. In particular:

1. An SSL certificate and key in `/data` and named `server.cert`, `server.key`
2. An `htpasswd` file at `/data/htpasswd`
3. The kubernetes apiserver is on `127.0.0.1:8080`

## Running the Container
Running the container can be done with a similar command to the following but assumes you've placed with ssl cert and htpasswd files in the corresponding locations.

    docker run --rm --net="host" -p "443:443" -t --name "kubernetes_nginx" -v "/opt/kubernetes_nginx:/data/:ro" kubernetes_nginx

## Helper Scripts
There are two helper scripts in this directory:

1. `make-cert.sh` -  This will generate a self signed certificate and key in the directories specified by arguements $1 and $2
2. `make-ca-cert.sh` - *currently GCE specific* - This script will build a CA certificate.

## Notes
-  `--net="host"` is being used since the kubernetes apiserver runs on the locally on the master node by default. This will vary by implementation.
