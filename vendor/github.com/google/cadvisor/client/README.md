# Example REST API Client

This is an implementation of a cAdvisor REST API in Go.  You can use it like this:

```go
client, err := client.NewClient("http://192.168.59.103:8080/")
```

Obviously, replace the URL with the path to your actual cAdvisor REST endpoint.


### MachineInfo

```go
client.MachineInfo()
```

This method returns a cadvisor/v1.MachineInfo struct with all the fields filled in.  Here is an example return value:

```
(*v1.MachineInfo)(0xc208022b10)({
 NumCores: (int) 4,
 MemoryCapacity: (int64) 2106028032,
 Filesystems: ([]v1.FsInfo) (len=1 cap=4) {
  (v1.FsInfo) {
   Device: (string) (len=9) "/dev/sda1",
   Capacity: (uint64) 19507089408
  }
 }
})
```

You can see the full specification of the [MachineInfo struct in the source](../info/v1/machine.go#L131)

### ContainerInfo

Given a container name and a [ContainerInfoRequest](../info/v1/container.go#L101), will return all information about the specified container.  See the [ContainerInfoRequest struct in the source](../info/v1/container.go#L101) for the full specification.

```go
request := v1.ContainerInfoRequest{NumStats: 10}
sInfo, err := client.ContainerInfo("/docker/d9d3eb10179e6f93a...", &request)
```
Returns a [ContainerInfo struct](../info/v1/container.go#L128)

### SubcontainersInfo

Given a container name and a [ContainerInfoRequest](../info/v1/container.go#L101), will recursively return all info about the container and all subcontainers contained within the container.  See the [ContainerInfoRequest struct in the source](../info/v1/container.go#L101) for the full specification.

```go
request := v1.ContainerInfoRequest{NumStats: 10}
sInfo, err := client.SubcontainersInfo("/docker", &request)
```

Returns a [ContainerInfo struct](../info/v1/container.go#L128) with the Subcontainers field populated.
