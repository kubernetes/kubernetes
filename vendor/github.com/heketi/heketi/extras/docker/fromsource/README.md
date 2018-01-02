# Overview
The main purpose of this container is to be used for testing
and verification of the unstable master builds.

# How to use for testing

## Downloading
First you will need to download the latest development container:

    # docker pull heketi/heketi:dev
    
> NOTE: Most likely you will always need to do a new pull before staring your tests since the container changes so often.

## Server Setup
You will need to create a directory which has a directory containing configuraiton and any private key if necessary, and an empty directory used for storing the database.  Directory and files must be read/write by user with id 1000 and if an ssh private key is used, it must also have a mod of 0600.

Here is an example:

    $ mkdir -p heketi/config
    $ mkdir -p heketi/db
    $ cp heketi.json heketi/config
    $ cp myprivate_key heketi/config
    $ chmod 600 heketi/config/myprivate_key
    $ chown 1000:1000 -R heketi

To run:

    # docker run -d -p 8080:8080 \
                 -v $PWD/heketi/config:/etc/heketi \
                 -v $PWD/heketi/db:/var/lib/heketi \
                 heketi/heketi:dev

Now you can see the container running.  Here is an example:

```
$ sudo docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
6e3ed5c59f87        heketidev           "/usr/bin/heketi -con"   32 minutes ago      Up 32 minutes       0.0.0.0:8080->8080/tcp   goofy_kowalevski
```

Now we can check the logs

```
$ sudo docker logs 6e3ed5c59f87 | head 
Heketi 1.0.0-81-g0c78700
[heketi] INFO 2016/04/12 18:57:13 Loaded ssh executor
[heketi] INFO 2016/04/12 18:57:13 Loaded simple allocator
[heketi] INFO 2016/04/12 18:57:13 GlusterFS Application Loaded
[negroni] Started GET /hello
[negroni] Completed 200 OK in 79.951µs
[negroni] Started GET /clusters
[negroni] Completed 200 OK in 91.658µs
[negroni] Started POST /clusters
[negroni] Completed 201 Created in 6.046309ms
```

## Using heketi-cli
Using our example above, to use the heketi-cli, you can type:

```
$ sudo docker exec 6e3ed5c59f87 \
    heketi-cli -h
$ sudo docker exec 6e3ed5c59f87 \
    heketi-cli --server http://localhost:8080/ cluster list
```

# Build
If you need to build it:

    # docker build --rm --tag <username>/heketi:dev .

