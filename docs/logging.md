# Logging

## Logging by Kubernetes Components
Kubernetes components, such as kubelet and apiserver, use the [glog](https://godoc.org/github.com/golang/glog) logging library.  Developer conventions for logging severity are described in [devel/logging.md](devel/logging.md).

## Logging in Containers
There are no Kubernetes-specific requirements for logging from within containers.  A
[search](https://www.google.com/?q=docker+container+logging) will turn up any number of articles about logging and
Docker containers.  However, we do provide an example of how to collect, index, and view pod logs [using Elasicsearch and Kibana](./getting-started-guides/logging.md)



