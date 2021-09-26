 #!/usr/bin/env pwsh -c

<#
.DESCRIPTION
Creates a GitHub pull request for a given branch if it doesn't already exist
.PARAMETER RepoOwner
The GitHub repository owner to create the pull request against.
.PARAMETER RepoName
The GitHub repository name to create the pull request against.
.PARAMETER BaseBranch
The base or target branch we want the pull request to be against.
.PARAMETER PROwner
The owner of the branch we want to create a pull request for.
.PARAMETER PRBranch
The branch which we want to create a pull request for.
.PARAMETER AuthToken
A personal access token
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)]
  $RepoOwner,

  [Parameter(Mandatory = $true)]
  $RepoName,

  [Parameter(Mandatory = $true)]
  $BaseBranch,

  [Parameter(Mandatory = $true)]
  $PROwner,

  [Parameter(Mandatory = $true)]
  $PRBranch,

  [Parameter(Mandatory = $true)]
  $AuthToken,

  [Parameter(Mandatory = $true)]
  $PRTitle,
  $PRBody = $PRTitle
)

$query = "state=open&head=${PROwner}:${PRBranch}&base=${BaseBranch}"

$resp = Invoke-RestMethod "https://api.github.com/repos/$RepoOwner/$RepoName/pulls?$query"
$resp | Write-Verbose

if ($resp.Count -gt 0) {
    Write-Host -f green "Pull request already exists $($resp[0].html_url)"
}
else {
  $headers = @{
    Authorization = "bearer $AuthToken"
  }

  $data = @{
    title                 = $PRTitle
    head                  = "${PROwner}:${PRBranch}"
    base                  = $BaseBranch
    body                  = $PRBody
    maintainer_can_modify = $true
  }

  $resp = Invoke-RestMethod -Method POST -Headers $headers `
                            https://api.github.com/repos/$RepoOwner/$RepoName/pulls `
                            -Body ($data | ConvertTo-Json)
  $resp | Write-Verbose
  Write-Host -f green "Pull request created https://github.com/$RepoOwner/$RepoName/pull/$($resp.number)"
}
