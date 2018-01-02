Enumerate Regions and Endpoints Example
===

Demonstrates how the SDK's endpoints can be enumerated over to discover regions, services, and endpoints defined by the SDK's Regions and Endpoints metadata.

Usage
---

The following parameters can be used to enumerate the SDK's partition metadata.

Example:

    go run -tags example enumEndpoints.go -p aws -services -r us-west-2

Output:

    Services with endpoint us-west-2 in aws:
	ec2
	dynamodb
	s3
    ...

CLI parameters
---

```
    -p=id partition id, e.g: aws
    -r=id region id, e.g: us-west-2
    -s=id service id, e.g: s3
    
    -partitions Lists all partitions.
    -regions Lists all regions in a partition. Requires partition ID.
             If service ID is also provided will show endpoints for a service.
    -services Lists all services in a partition. Requires partition ID.
              If region ID is also provided, will show services available in that region.
```

