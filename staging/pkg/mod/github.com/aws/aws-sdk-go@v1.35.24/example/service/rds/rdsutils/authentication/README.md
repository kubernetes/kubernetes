# Example

This is an example using the AWS SDK for Go to create an Amazon RDS DB token using the
rdsutils package.

# Usage

```sh
go run -tags example iam_authentication.go <region> <db user> <db name> <endpoint to database> <iam arn>
```

Output:
```
Successfully opened connection to database
```
