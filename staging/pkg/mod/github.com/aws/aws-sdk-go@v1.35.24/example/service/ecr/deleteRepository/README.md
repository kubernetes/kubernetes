# Example

This is an example using the AWS SDK for Go to delete an ECR Repository.


# Usage

The example deletes the repository name provided in the first parameter from default `us-east-1` aws region.

```sh
go run -tags example deleteRepository.go <repo_name>
```

To delete a repository from a different region, set `AWS_REGION` env var.

```sh
AWS_REGION=us-west-2 go run -tags example deleteRepository.go <repo_name>
```

Output:
```
ECR Repository "repo2" deleted successfully!

AWS Output:
{
  Repository: {
    CreatedAt: 2020-03-26 00:20:36 +0000 UTC,
    RegistryId: "999999999999",
    RepositoryArn: "arn:aws:ecr:us-east-1:999999999999:repository/repo_name",
    RepositoryName: "repo_name",
    RepositoryUri: "999999999999.dkr.ecr.us-east-1.amazonaws.com/repo_name"
  }
}
```
