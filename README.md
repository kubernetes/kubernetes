# Kubernetes

[![Submit Queue Widget]][Submit Queue] [![GoDoc Widget]][GoDoc] [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/569/badge)](https://bestpractices.coreinfrastructure.org/projects/569)

<img src="https://github.com/kubernetes/kubernetes/raw/master/logo/logo.png" width="100">

----

Kubernetes is a canonical, open source implementation of a container orchestration system for running microservices in containers, across multiple hosts, with provisioning basic mechanisms for end to end deployment, scheduling, scaling, and networking of microservices, with examples of similar systems, you can check out Mesos, Docker Swarm, Nomad.

Kubernetes builds upon 17 years of experience at Google running
production workloads at scale using a system called [Borg],
combined with best-of-breed ideas and practices from the community.

Kubernetes is hosted by the Cloud Native Computing Foundation ([CNCF]).
If you are a company that wants to help shape the evolution of
technologies that are container-packaged, dynamically-scheduled
and microservices-oriented, consider joining the CNCF.
For details about who's involved and how Kubernetes plays a role,
read the CNCF [announcement].

----

## Getting started with Kubernetes as a User

See our documentation on [kubernetes.io].

Try our [interactive tutorial].

Take a free course on [Scalable Microservices with Kubernetes].

## Getting started with Kubernetes as a Developer

The [community repository] hosts all information about
building Kubernetes from source, contributing code
and documentation, what individuals/SIG's contact for specific
questions.

If you want to build Kubernetes right away there are two options:

### If you have a working [Go environment]:

```
$ go get -d k8s.io/kubernetes
$ cd $GOPATH/src/k8s.io/kubernetes
$ make
```

## If you have a working [Docker environment]:

```
$ git clone https://github.com/kubernetes/kubernetes
$ cd kubernetes
$ make quick-release
```

For the full story, head over to the [developer's documentation].

## Support

If you need support, start with the [troubleshooting guide],
and work your way through the process that we've outlined.

That said, if you have questions, reach out to us
[one way or another][communication].

[announcement]: https://cncf.io/news/announcement/2015/07/new-cloud-native-computing-foundation-drive-alignment-among-container
[Borg]: https://research.google.com/pubs/pub43438.html
[CNCF]: https://www.cncf.io/about
[communication]: https://git.k8s.io/community/communication
[community repository]: https://git.k8s.io/community
[containerized applications]: https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/
[developer's documentation]: https://git.k8s.io/community/contributors/devel#readme
[Docker environment]: https://docs.docker.com/engine
[Go environment]: https://golang.org/doc/install
[GoDoc]: https://godoc.org/k8s.io/kubernetes
[GoDoc Widget]: https://godoc.org/k8s.io/kubernetes?status.svg
[interactive tutorial]: http://kubernetes.io/docs/tutorials/kubernetes-basics
[kubernetes.io]: http://kubernetes.io
[Scalable Microservices with Kubernetes]: https://www.udacity.com/course/scalable-microservices-with-kubernetes--ud615
[Submit Queue]: http://submit-queue.k8s.io/#/ci
[Submit Queue Widget]: http://submit-queue.k8s.io/health.svg?v=1
[troubleshooting guide]: https://kubernetes.io/docs/tasks/debug-application-cluster/troubleshooting/

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/README.md?pixel)]()
