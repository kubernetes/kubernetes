#Requires -Version 6.0

<#
.SYNOPSIS
This script will do the necessary book keeping work needed to release a package.

.DESCRIPTION
This script will do a number of things when ran:

- It will read the current version from the project and will have you confirm if that is the version you want to ship
- It will take the package metadata and version and update the DevOps release tracking items with that information.
  - If there is existing release work item it will update it and if not it will create one.
- It will validate that the changelog has a entry for the package version that you want to release as well as a timestamp.

.PARAMETER PackageName
The full package name of the package you want to prepare for release. (i.e Azure.Core, azure-core, @azure/core-https)

.PARAMETER ServiceDirectory
Optional: The service directory where the package lives (e.g. /sdk/<service directory>/<package>). If a service directory isn't provided the script
will search for the package project by traversing all the packages under /sdk/, so the service directory is only a scoping mechanism.

.PARAMETER ReleaseDate
Optional: If not shipping on the normal first Tuesday of the month you can specify a specific release date in the form of "MM/dd/yyyy".

.PARAMETER ReleaseTrackingOnly
Optional: If this switch is passed then the script will only update the release work items and not update the versions in the local repo or validate the changelog.

.EXAMPLE
PS> ./eng/common/scripts/Prepare-Release.ps1 <PackageName>

The most common usage is to call the script passing the package name. Once the script is finished then you will have modified project and change log files.
You should make any additional changes to the change log to capture the changes and then submit the PR for the final changes before you do a release.

.EXAMPLE
PS> ./eng/common/scripts/Prepare-Release.ps1 <PackageName> -ReleaseTrackingOnly

If you aren't ready to do the final versioning changes yet but you want to update release tracking information for shiproom pass in the -ReleaseTrackingOnly.
option. This should not modify or validate anything in the repo but will update the DevOps release tracking items. Once you are ready for the verioning changes
as well then come back and run the full script again without the -ReleaseTrackingOnly option and give it the same version information you did the first time.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$PackageName,
  [string]$ServiceDirectory,
  [string]$ReleaseDate, # Pass Date in the form MM/dd/yyyy"
  [switch]$ReleaseTrackingOnly = $false
)
Set-StrictMode -Version 3

. ${PSScriptRoot}\common.ps1

function Get-ReleaseDay($baseDate)
{
  # Find first friday
  while ($baseDate.DayOfWeek -ne 5)
  {
    $baseDate = $baseDate.AddDays(1)
  }

  # Go to Tuesday
  $baseDate = $baseDate.AddDays(4)

  return $baseDate;
}

$ErrorPreference = 'Stop'

$packageProperties = $null
$packageProperties = Get-PkgProperties -PackageName $PackageName -ServiceDirectory $ServiceDirectory

if (!$packageProperties)
{
  Write-Error "Could not find a package with name [ $packageName ], please verify the package name matches the exact name."
  exit 1
}

Write-Host "Package Name [ $($packageProperties.Name) ]"
Write-Host "Source directory [ $($packageProperties.ServiceDirectory) ]"

if (!$ReleaseDate)
{
  $currentDate = Get-Date
  $thisMonthReleaseDate = Get-ReleaseDay((Get-Date -Day 1));
  $nextMonthReleaseDate = Get-ReleaseDay((Get-Date -Day 1).AddMonths(1));

  if ($thisMonthReleaseDate -ge $currentDate)
  {
    # On track for this month release
    $ParsedReleaseDate = $thisMonthReleaseDate
  }
  elseif ($currentDate.Day -lt 15)
  {
    # Catching up to this month release
    $ParsedReleaseDate = $currentDate
  }
  else
  {
    # Next month release
    $ParsedReleaseDate = $nextMonthReleaseDate
  }
}
else
{
  $ParsedReleaseDate = [datetime]$ReleaseDate
}

$releaseDateString = $ParsedReleaseDate.ToString("MM/dd/yyyy")
$month = $ParsedReleaseDate.ToString("MMMM")

Write-Host
Write-Host "Assuming release is in $month with release date $releaseDateString" -ForegroundColor Green

$currentProjectVersion = $packageProperties.Version

$newVersion = Read-Host -Prompt "Input the new version, or press Enter to use use current project version '$currentProjectVersion'"

if (!$newVersion)
{
  $newVersion = $currentProjectVersion;
}

$newVersionParsed = [AzureEngSemanticVersion]::ParseVersionString($newVersion)
if ($null -eq $newVersionParsed)
{
  Write-Error "Invalid version $newVersion. Version must follow standard SemVer rules, see https://aka.ms/azsdk/engsys/packageversioning"
  exit 1
}

&$EngCommonScriptsDir/Update-DevOps-Release-WorkItem.ps1 `
  -language $LanguageDisplayName `
  -packageName $packageProperties.Name `
  -version $newVersion `
  -plannedDate $releaseDateString `
  -packageRepoPath $packageProperties.serviceDirectory `
  -packageType $packageProperties.SDKType `
  -packageNewLibrary $packageProperties.IsNewSDK

if ($LASTEXITCODE -ne 0) {
  Write-Error "Updating of the Devops Release WorkItem failed."
  exit 1
}

if ($releaseTrackingOnly)
{
  Write-Host
  Write-Host "Script is running in release tracking only mode so only updating the release tracker and not updating versions locally."
  Write-Host "You will need to run this script again once you are ready to update the versions to ensure the projects and changelogs contain the correct version."

  exit 0
}

if (Test-Path "Function:SetPackageVersion")
{
  SetPackageVersion -PackageName $packageProperties.Name -Version $newVersion -ServiceDirectory $packageProperties.ServiceDirectory -ReleaseDate $releaseDateString `
    -PackageProperties $packageProperties
}
else
{
  LogError "The function 'SetPackageVersion' was not found.`
    Make sure it is present in eng/scripts/Language-Settings.ps1.`
    See https://github.com/Azure/azure-sdk-tools/blob/master/doc/common/common_engsys.md#code-structure"
  exit 1
}

git diff -s --exit-code $packageProperties.DirectoryPath
if ($LASTEXITCODE -ne 0)
{
  git status
  Write-Host "Some changes were made to the repo source" -ForegroundColor Green
  Write-Host "Submit a pull request with the necessary changes to the repo, including any final changelog entry updates." -ForegroundColor Green
}
