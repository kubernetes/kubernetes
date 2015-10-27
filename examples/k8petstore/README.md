<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Welcome to k8PetStore

This is a follow up to the [Guestbook Example](../guestbook/README.md)'s [Go implementation](../guestbook-go/).

- It leverages the same components (redis, Go REST API) as the guestbook application
- It comes with visualizations for graphing whats happening in Redis transactions, along with commandline printouts of transaction throughput
- It is hackable : you can build all images from the files is in this repository (With the exception of the data generator, which is apache bigtop).
- It generates massive load using a semantically rich, realistic transaction simulator for petstores

This application will run a web server which returns REDIS records for a petstore application.
It is meant to simulate and test high load on Kubernetes or any other docker based system.

If you are new to Kubernetes, and you haven't run guestbook yet,

you might want to stop here and go back and run guestbook app first.

The guestbook tutorial will teach you a lot about the basics of Kubernetes, and we've tried not to be redundant here.

## Architecture of this SOA

A diagram of the overall architecture of this application can be seen in [k8petstore.dot](k8petstore.dot) (you can paste the contents in any graphviz viewer, including online ones such as http://sandbox.kidstrythisathome.com/erdos/.

## Docker image dependencies

Reading this section is optional, only if you want to rebuild everything from scratch.

This project depends on three docker images which you can build for yourself and save
in your dockerhub "dockerhub-name".

Since these images are already published under other parties like redis, jayunit100, and so on,
so you don't need to build the images to run the app.

If you do want to build the images, you will need to build and push the images in this repository.

For a list of those images, see the `build-and-push` shell script - it builds and pushes all the images for you, just

modify the dockerhub user name in it accordingly.

## Get started with the WEBAPP

The web app is written in Go, and borrowed from the original Guestbook example by brendan burns.

We have extended it to do some error reporting, persisting of JSON petstore transactions (not much different then guestbook entries),

and supporting of additional REST calls, like LLEN, which returns the total # of transactions in the database.

To work on the app, just cd to the `dev` directory, and follow the instructions.  You can easily edit it in your local machine, by installing

redis and go.  Then you can use the `Vagrantfile` in this top level directory to launch a minimal version of the app in pure docker containers.

If that is all working, you can finally run `k8petstore.sh` in any Kubernetes cluster, and run the app at scale.

## Set up the data generator (optional)

The web front end provides users an interface for watching pet store transactions in real time as they occur.

To generate those transactions, you can use the bigpetstore data generator.  Alternatively, you could just write a

shell script which calls "curl localhost:3000/k8petstore/rpush/blahblahblah" over and over again :).  But that's not nearly

as fun, and its not a good test of a real world scenario where payloads scale and have lots of information content.

Similarly, you can locally run and test the data generator code, which is Java based, you can pull it down directly from

apache bigtop.

Directions for that are here : https://github.com/apache/bigtop/tree/master/bigtop-bigpetstore/bigpetstore-transaction-queue

You will likely want to checkout the branch 2b2392bf135e9f1256bd0b930f05ae5aef8bbdcb, which is the exact commit which the current k8petstore was tested on.

## Now what?

Once you have done the above 3 steps, you have a working, from source, locally runnable version of the k8petstore app, now, we can try to run it in Kubernetes.

## Hacking, testing, benchmarking

Once the app is running, you can access the app in your browser, you should see a chart

and the k8petstore title page, as well as an indicator of transaction throughput, and so on.

You can modify the HTML pages, add new REST paths to the Go app, and so on.

## Running in Kubernetes

Now that you are done hacking around on the app, you can run it in Kubernetes.  To do this, you will want to rebuild the docker images (most likely, for the Go web-server app), but less likely for the other images which you are less likely to need to change. Then you will push those images to dockerhub.

Now, how to run the entire application in Kubernetes?

To simplify running this application, we have a single file, k8petstore.sh, which writes out json files on to disk.  This allows us to have dynamic parameters, without needing to worry about managing multiple json files.

You might want to change it to point to your customized Go image, if you chose to modify things, like the number of data generators (more generators will create more load on the redis master).

So, to run this app in Kubernetes, simply run [The all in one k8petstore.sh shell script](k8petstore.sh).

Note that at the top of the script there are a few self explanatory parameters to set, among which the Public IPs parameter is where you can checkout the web ui (at $PUBLIC_IP:3000), which will show a plot and read outs of transaction throughput.

In the mean time, because the public IP will be deprecated in Kubernetes v1, we provide other 2 scripts k8petstore-loadbalancer.sh and k8petstore-nodeport.sh. As the names suggest, they rely on LoadBalancer and NodePort respectively. More details can be found [here](../../docs/user-guide/services.md#external-services).

## Future

In the future, we plan to add cassandra support.  Redis is a fabulous in memory data store, but it is not meant for truly available and resilient storage.

Thus we plan to add another tier of queueing, which empties the REDIS transactions into a cassandra store which persists.

## Questions

For questions on running this app, you can ask on [Slack](../../docs/troubleshooting.md#slack).

For questions about bigpetstore, and how the data is generated, ask on the apache bigtop mailing list.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/k8petstore/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
