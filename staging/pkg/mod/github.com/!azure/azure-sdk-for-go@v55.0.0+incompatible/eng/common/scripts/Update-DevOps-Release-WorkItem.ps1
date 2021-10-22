
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$language,
  [Parameter(Mandatory=$true)]
  [string]$packageName,
  [Parameter(Mandatory=$true)]
  [string]$version,
  [string]$plannedDate,
  [string]$serviceName = $null,
  [string]$packageDisplayName = $null,
  [string]$packageRepoPath = "NA",
  [string]$packageType = "client",
  [string]$packageNewLibrary = "true",
  [string]$devops_pat = $env:DEVOPS_PAT
)
#Requires -Version 6.0
Set-StrictMode -Version 3

if (!(Get-Command az -ErrorAction SilentlyContinue)) {
  Write-Error 'You must have the Azure CLI installed: https://aka.ms/azure-cli'
  exit 1
}

az account show *> $null
if (!$?) {
  Write-Host 'Running az login...'
  az login *> $null
}

az extension show -n azure-devops *> $null
if (!$?){
  Write-Host 'Installing azure-devops extension'
  az extension add --name azure-devops
}

. (Join-Path $PSScriptRoot SemVer.ps1)
. (Join-Path $PSScriptRoot Helpers DevOps-WorkItem-Helpers.ps1)

$parsedNewVersion = [AzureEngSemanticVersion]::new($version)
$state = "In Release"
$releaseType = $parsedNewVersion.VersionType
$versionMajorMinor = "" + $parsedNewVersion.Major + "." + $parsedNewVersion.Minor

$packageInfo = [PSCustomObject][ordered]@{
  Package = $packageName
  DisplayName = $packageDisplayName
  ServiceName = $serviceName
  RepoPath = $packageRepoPath
  Type = $packageType
  New = $packageNewLibrary
};

if (!$plannedDate) {
  $plannedDate = Get-Date -Format "MM/dd/yyyy"
}

$plannedVersions = @(
  [PSCustomObject][ordered]@{
    Type = $releaseType
    Version = $version
    Date = $plannedDate
  }
)

$workItem = FindOrCreateClonePackageWorkItem $language $packageInfo $versionMajorMinor -allowPrompt $true -outputCommand $false

if (!$workItem) {
  Write-Host "Something failed as we don't have a work-item so exiting."
  exit 1
}

Write-Host "Updated or created a release work item for a package release with the following properties:"
Write-Host "  Lanuage: $($workItem.fields['Custom.Language'])"
Write-Host "  Version: $($workItem.fields['Custom.PackageVersionMajorMinor'])"
Write-Host "  Package: $($workItem.fields['Custom.Package'])"
Write-Host "  AssignedTo: $($workItem.fields['System.AssignedTo']["uniqueName"])"
Write-Host "  PackageDisplayName: $($workItem.fields['Custom.PackageDisplayName'])"
Write-Host "  ServiceName: $($workItem.fields['Custom.ServiceName'])"
Write-Host "  PackageType: $($workItem.fields['Custom.PackageType'])"
Write-Host ""
Write-Host "Marking item [$($workItem.id)]$($workItem.fields['System.Title']) as '$state' for '$releaseType'"
$updatedWI = UpdatePackageWorkItemReleaseState -id $workItem.id -state "In Release" -releaseType $releaseType -outputCommand $false
$updatedWI = UpdatePackageVersions $workItem -plannedVersions $plannedVersions

Write-Host "Release tracking item is at https://dev.azure.com/azure-sdk/Release/_workitems/edit/$($updatedWI.id)/"
