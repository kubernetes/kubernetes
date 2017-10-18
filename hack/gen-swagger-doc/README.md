This folder contains the sources needed to build the gen-swagger-doc container.

To build the container image, 

```
$ sudo docker build -t k8s.gcr.io/gen-swagger-docs:v1 .
```

To generate the html docs,

```
$ ./gen-swagger-docs.sh <API version> <absolute output path, default to PWD>
```

The generated definitions.html and operations.html will be stored in output paths.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/hack/gen-swagger-doc/README.md?pixel)]()
