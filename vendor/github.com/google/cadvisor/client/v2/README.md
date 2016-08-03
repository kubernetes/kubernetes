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

There is no v2 MachineInfo API, so the v2 client exposes the [v1 MachineInfo](../../info/v1/machine.go#L131)

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

You can see the full specification of the [MachineInfo struct in the source](../../info/v1/machine.go#L131)

### VersionInfo

```go
client.VersionInfo()
```

This method returns the cAdvisor version.

### Attributes

```go
client.Attributes()
```

This method returns a [cadvisor/info/v2/Attributes](../../info/v2/machine.go#L24) struct with all the fields filled in. Attributes includes hardware attributes (as returned by MachineInfo) as well as software attributes (eg. software versions). Here is an example return value:

```
(*v2.Attributes)({
 KernelVersion: (string) (len=17) "3.13.0-44-generic"
 ContainerOsVersion: (string) (len=18) "Ubuntu 14.04.1 LTS"
 DockerVersion: (string) (len=9) "1.5.0-rc4"
 CadvisorVersion: (string) (len=6) "0.10.1"
 NumCores: (int) 4,
 MemoryCapacity: (int64) 2106028032,
 Filesystems: ([]v2.FsInfo) (len=1 cap=4) {
  (v2.FsInfo) {
   Device: (string) (len=9) "/dev/sda1",
   Capacity: (uint64) 19507089408
  }
 }
})
```

You can see the full specification of the [Attributes struct in the source](../../info/v2/machine.go#L24)

