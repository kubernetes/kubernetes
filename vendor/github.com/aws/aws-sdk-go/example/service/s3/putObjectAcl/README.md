# Example

putObjectAcl is an example using the AWS SDK for Go to put an ACL on an S3 object.

# Usage

```sh
putBucketAcl <params>
	-region <region> // required
	-bucket <bucket> // required
	-key <key> // required
	-owner-name <owner-name>
	-owner-id <owner-id>
	-grantee-type <some type> // required
	-uri <uri to group>
	-email <email address>
	-user-id <user-id>
	-display-name <display name>
```

```sh
go run -tags example putObjectAcl.go 
	-bucket <bucket> 
	-key <key> 
	-owner-name <name> 
	-owner-id <id>
	-grantee-type <some type>
	-user-id <user-id>
```

Depending on the type is used depends on which of the three, `uri`, `email`, or `user-id`, needs to be used.
* `s3.TypeCanonicalUser`: `user-id` or `display-name` must be used
* `s3.TypeAmazonCustomerByEmail`: `email` must be used
* `s3.TypeGroup`: `uri` must be used

Output:
```
success {
} nil
```
