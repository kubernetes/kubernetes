param(
  $Repository,
  $Tag,
  $AuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

$repositoryParts = $Repository.Split("/")

if ($repositoryParts.Length -ne 2)
{
    LogError "Repository is not a valid format."
}

$repositoryOwner = $repositoryParts[0]
LogDebug "Repository owner is: $repositoryOwner"

$repositoryName = $repositoryParts[1]
LogDebug "Reposiory name is: $repositoryName"

$ref = "tags/$Tag"
LogDebug "Calculated ref is: $ref"

try
{
    Remove-GitHubSourceReferences -RepoOwner $repositoryOwner -RepoName $repositoryName -Ref $ref -AuthToken $AuthToken
}
catch
{
  LogError "Remove-GitHubSourceReferences failed with exception:`n$_"
  exit 1
}