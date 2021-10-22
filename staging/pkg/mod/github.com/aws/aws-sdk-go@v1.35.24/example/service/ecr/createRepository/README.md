# Example

This is an example using the AWS SDK for Go to create a new ECR Repository.


# Usage

The example creates the repository name provided in the first parameter on default `us-east-1` aws region.

```sh
go run -tags example createRepository.go <repo_name>
```

To create a repository in a different region, set `AWS_REGION` env var.

```sh
AWS_REGION=us-west-2 go -tags example run createRepository.go <repo_name>
```

Output:
```
ECR Repository "repo_name" created successfully!

AWS Output:
{
  Repository: {
    CreatedAt: 2020-03-26 00:20:36 +0000 UTC,
    ImageScanningConfiguration: {
      ScanOnPush: false
    },
    ImageTagMutability: "MUTABLE",
    RegistryId: "999999999999",
    RepositoryArn: "arn:aws:ecr:us-east-1:999999999999:repository/repo_name",
    RepositoryName: "repo_name",
    RepositoryUri: "999999999999.dkr.ecr.us-east-1.amazonaws.com/repo_name"
  }
}
```
