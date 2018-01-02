# Example

This is an example using the AWS SDK for Go to list ec2 instances that match provided tag name filter.


# Usage

The example uses the bucket name provided, and lists all object keys in a bucket.

```sh
go run -tags example filter_ec2_by_tag.go <name_filter>
```

Output:
```
listing instances with tag vpn in: us-east-1
[{
  Instances: [{
      AmiLaunchIndex: 0,
      Architecture: "x86_64",
      BlockDeviceMappings: [{
          DeviceName: "/dev/xvda",
          Ebs: {
            AttachTime: 2016-07-06 18:04:53 +0000 UTC,
            DeleteOnTermination: true,
            Status: "attached",
            VolumeId: "vol-xxxx"
          }
        }],
      ...
```
