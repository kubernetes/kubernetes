# Intended to be used at the beginning of CI process to easily encapsulate the work of creating a new Go workspace.

# On completion. Returns two variables in a PSObject. 
#  GO_WORKSPACE_PATH <- location of copied sources directory
#  GO_PATH <- The value that should be set for the GO_PATH environment variable
Param(
  [string] $goWorkSpaceDir,
  [string] $orgOrUser = "Azure",
  [string] $repo = "azure-sdk-for-go"
)
$repoRoot = Resolve-Path "$PSScriptRoot/../../"

$CreatedGoWorkspaceSrc = "$goWorkSpaceDir/src/github.com/$orgOrUser/$repo/"
$CreatedGoWorkspacePkg = "$goWorkSpaceDir/pkg"

# create base two folders for the root of the go workspace
New-Item -ItemType Directory -Force -Path $CreatedGoWorkspaceSrc
New-Item -ItemType Directory -Force -Path $CreatedGoWorkspacePkg

Write-Host "Source is $repoRoot"
Write-Host "Destination is $CreatedGoWorkspaceSrc"
Write-Host "Root of new Go Workspace is $goWorkSpaceDir"

Copy-Item -Container -Recurse -Path "$repoRoot/*" -Destination $CreatedGoWorkspaceSrc

return New-Object PSObject -Property @{
  GO_WORKSPACE_PATH = Resolve-Path $CreatedGoWorkspaceSrc
  GO_PATH = Resolve-Path $goWorkSpaceDir
}
