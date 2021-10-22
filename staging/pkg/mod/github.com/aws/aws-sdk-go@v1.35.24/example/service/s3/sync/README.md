# Example

sync will upload a given directory to Amazon S3 using the upload iterator interface defined in the
s3manager package. This example uses a path that is specified during runtime to walk and build keys
to upload to Amazon S3. It will use the keys to upload the files/folders to Amazon S3.

# Usage

```sh
sync <params>
	-region <region> // required
	-bucket <bucket> // required
	-path  <path> // required
```

```sh
go run -tags example sync.go
	-region <region> // required
	-bucket <bucket> // required
	-path  <path> // required
```

Output:
```
success
```
