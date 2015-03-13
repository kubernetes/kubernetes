## Welcome to k8PetStore

This is a follow up to the Guestbook example, which implements a slightly more real world demonstration using 

the same application architecture.

- It leverages the same components (redis, Go REST API) as the guestbook application
- It comes with visualizations for graphing whats happening in Redis transactions, along with commandline printouts of transaction throughput
- It is hackable : you can build all images from the files is in this repository (With the exception of the data generator, which is apache bigtop).
- It generates massive load using a semantically rich, realistic transaction simulator for petstores

This application will run a web server which returns REDIS records for a petstore application.
It is meant to simulate and test high load on kubernetes or any other docker based system.

If you are new to kubernetes, and you haven't run guestbook yet, 

you might want to stop here and go back and run guestbook app first.  

The guestbook tutorial will teach you alot about the basics of kubernetes, and we've tried not to be redundant here.

## Architecture of this SOA

A diagram of the overall architecture of this application can be seen in arch.dot (you can paste the contents in any graphviz viewer, including online ones such as http://sandbox.kidstrythisathome.com/erdos/.

## Docker image dependencies

Reading this section is optional, only if you want to rebuild everything from scratch.

This project depends on three docker images which you can build for yourself and save
in your dockerhub "dockerhub-name".

Since these images are already published under other parties like redis, jayunit100, and so on,
so you don't need to build the images to run the app. 

If you do want to build the images, you will need to build and push these 3 docker images.

- dockerhub-name/bigpetstore-load-generator, which generates transactions for the database.
- dockerhub-name/redis, which is a simple curated redis image.
- dockerhub-name/k8petstore, which is the web app image.

## Get started with the WEBAPP 

The web app is written in Go, and borrowed from the original Guestbook example by brendan burns.

We have extended it to do some error reporting, persisting of JSON petstore transactions (not much different then guestbook entries),

and supporting of additional REST calls, like LLEN, which returns the total # of transactions in the database.

To run it locally, you simply need to run basic Go commands.  Assuming you have Go set up, do something like: 

```
#Assuming your gopath is in / (i.e. this is the case, for example, in our Dockerfile).
go get main
go build main
export STATIC_FILES=/tmp/static
/gopath/bin/main
```

## Set up the data generator

The web front end provides users an interface for watching pet store transactions in real time as they occur.

To generate those transactions, you can use the bigpetstore data generator.  Alternatively, you could just write a 

shell script which calls "curl localhost:3000/k8petstore/rpush/blahblahblah" over and over again :).  But thats not nearly

as fun, and its not a good test of a real world scenario where payloads scale and have lots of information content. 

Similarly, you can locally run and test the data generator code, which is Java based, you can pull it down directly from 

apache bigtop.

Directions for that are here : https://github.com/apache/bigtop/tree/master/bigtop-bigpetstore/bigpetstore-transaction-queue

You will likely want to checkout the branch 2b2392bf135e9f1256bd0b930f05ae5aef8bbdcb, which is the exact commit which the current k8petstore was tested on.

## Set up REDIS

Install and run redis locally.  This can be done very easily on any Unix system, and redis starts in an insecure mode so its easy 

to develop against.

Install the bigpetstore-transaction-queue generator app locally (optional), but for realistic testing.
Then, run the go app directly.  You will have to get dependencies using go the first time (will add directions later for that, its easy).

## Now what? 

Once you have done the above 3 steps, you have a working, from source, locally runnable version of the k8petstore app, now, we can try to run it in kubernetes.

## Hacking, testing, benchmarking

Once the app is running, you can go to the location of publicIP:3000 (the first parameter in the script).  In your browser, you should see a chart 

and the k8petstore title page, as well as an indicator of transaction throughput, and so on.  You should be able to modify  

You can modify the HTML pages, add new REST paths to the Go app, and so on.

## Running in kubernetes

Now that you are done hacking around on the app, you can run it in kubernetes.  To do this, you will want to rebuild the docker images (most likely, for the Go web-server app), but less likely for the other images which you are less likely to need to change. Then you will push those images to dockerhub.

Now, how to run the entire application in kubernetes? 

To simplify running this application, we have a single file, k8petstore.sh, which writes out json files on to disk.  This allows us to have dynamic parameters, without needing to worry about managing multiplejson files.

You might want to change it to point to your customized Go image, if you chose to modify things.  

like the number of data generators (more generators will create more load on the redis master).

So, to run this app in kubernetes, simply run `k8petstore.sh`.

## Future

In the future, we plan to add cassandra support.  Redis is a fabulous in memory data store, but it is not meant for truly available and resilient storage.  

Thus we plan to add another tier of queueing, which empties the REDIS transactions into a cassandra store which persists.   

## Questions

For questions on running this app, you can ask on the google containers group.

For questions about bigpetstore, and how the data is generated, ask on the apache bigtop mailing list.

