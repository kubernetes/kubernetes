# Example

This is an example using the AWS SDK for Go to download an S3 object with a
progress bar.

# Usage

The example uses the bucket name provided, one key for object, and output the
progress to stdout.

```prompt
AWS_PROFILE=my-profile AWS_REGION=us-west-2 go run -tags example getObjectWithProgress.go cool-bucket my/object/prefix/cool_thing.zip

2019/02/22 13:04:52 File size is: 35.9 MB
2019/02/22 13:04:53 File size:35943530 downloaded:8580 percentage:0%
2019/02/22 13:04:53 File size:35943530 downloaded:17580 percentage:0%
2019/02/22 13:04:53 File size:35943530 downloaded:33940 percentage:0%
2019/02/22 13:04:53 File size:35943530 downloaded:34988 percentage:0%
2019/02/22 13:04:53 File size:35943530 downloaded:51348 percentage:0%
2019/02/22 13:04:53 File size:35943530 downloaded:52396 percentage:0%
...
```
