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

This method returns a cadvisor/info.MachineInfo struct with all the fields filled in.  Here is an example return value:

```
(*info.MachineInfo)(0xc208022b10)({
 NumCores: (int) 4,
 MemoryCapacity: (int64) 2106028032,
 Filesystems: ([]info.FsInfo) (len=1 cap=4) {
  (info.FsInfo) {
   Device: (string) (len=9) "/dev/sda1",
   Capacity: (uint64) 19507089408
  }
 }
})
```

You can see the full specification of the [MachineInfo struct in the source](../info/container.go)

### ContainerInfo

Given a container name and a ContainerInfoRequest, will return all information about the specified container.  The ContainerInfoRequest struct just has one field, NumStats, which is the number of stat entries that you want returned.

```go
request := info.ContainerInfoRequest{10}
sInfo, err := client.ContainerInfo("/docker/d9d3eb10179e6f93a...", &request)
```
Returns a [ContainerInfo struct](../info/container.go)

### SubcontainersInfo

Given a container name and a ContainerInfoRequest, will recursively return all info about the container and all subcontainers contained within the container.  The ContainerInfoRequest struct just has one field, NumStats, which is the number of stat entries that you want returned.

```go
request := info.ContainerInfoRequest{10}
sInfo, err := client.SubcontainersInfo("/docker", &request)
```

Returns a [ContainerInfo struct](../info/container.go) with the Subcontainers field populated.
