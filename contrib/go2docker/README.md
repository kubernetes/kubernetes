# go2docker

## Description

`go2docker` is a command line tool to create minimal docker images from
`SCRATCH` for go packages.

It is based on the [Docker Image Specification v1.0.0](https://github.com/docker/docker/blob/master/image/spec/v1.md).

## Usage
```
go2docker [-image NAMESPACE/BASENAME] [PACKAGES]
```

### Options
- `image`: namespace/name for the repository, default to go2docker/$(basename)

### Examples
```
$ go get -d github.com/golang/example/hello
$ go2docker -image golang/hello github.com/golang/example/hello | docker load
$ docker images | grep hello
golang/hello	   latest	e96b9f048cdf			2 seconds ago	1.477 MB
$ docker run golang/hello
Hello, Go examples!
```

## TODOs
- [ ] add command line flag for entrypoint
- [ ] add command line flag for exposed port
- [ ] add command line flag for volume
- [ ] go get the package if not present in `$GOPATH`
- [ ] add push command
- [ ] test more complicated package (ex: etcd)
- [ ] fix permission inside the tar

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/go2docker/README.md?pixel)]()
