$RepoRoot = Resolve-Path "${PSScriptRoot}..\..\..\.."
$EngDir = Join-Path $RepoRoot "eng"
$EngCommonDir = Join-Path $EngDir "common"
$EngCommonScriptsDir = Join-Path $EngCommonDir "scripts"
$EngScriptsDir = Join-Path $EngDir "scripts"

# Import required scripts
. (Join-Path $EngCommonScriptsDir SemVer.ps1)
. (Join-Path $EngCommonScriptsDir ChangeLog-Operations.ps1)
. (Join-Path $EngCommonScriptsDir Package-Properties.ps1)
. (Join-Path $EngCommonScriptsDir logging.ps1)
. (Join-Path $EngCommonScriptsDir Invoke-GitHubAPI.ps1)
. (Join-Path $EngCommonScriptsDir Invoke-DevOpsAPI.ps1)
. (Join-Path $EngCommonScriptsDir artifact-metadata-parsing.ps1)

# Setting expected from common languages settings
$Language = "Unknown"
$PackageRepository = "Unknown"
$packagePattern = "Unknown"
$MetadataUri = "Unknown"

# Import common language settings
$EngScriptsLanguageSettings = Join-path $EngScriptsDir "Language-Settings.ps1"
if (Test-Path $EngScriptsLanguageSettings) {
  . $EngScriptsLanguageSettings
}

if (!(Get-Variable -Name "LanguageShort" -ValueOnly -ErrorAction "Ignore"))
{
  $LanguageShort = $Language
}

if (!(Get-Variable -Name "LanguageDisplayName" -ValueOnly -ErrorAction "Ignore"))
{
  $LanguageDisplayName = $Language
}

# Transformed Functions
$GetPackageInfoFromRepoFn = "Get-${Language}-PackageInfoFromRepo"
$GetPackageInfoFromPackageFileFn = "Get-${Language}-PackageInfoFromPackageFile"
$PublishGithubIODocsFn = "Publish-${Language}-GithubIODocs"
$UpdateDocsMsPackagesFn = "Update-${Language}-DocsMsPackages"
$GetGithubIoDocIndexFn = "Get-${Language}-GithubIoDocIndex"
$FindArtifactForApiReviewFn = "Find-${Language}-Artifacts-For-Apireview"
