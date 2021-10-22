# Note: This script will add or replace version title in change log

# Parameter description
# Version : Version to add or replace in change log
# Unreleased: Default is true. If it is set to false, then today's date will be set in verion title. If it is True then title will show "Unreleased"
# ReplaceLatestEntryTitle: Replaces the latest changelog entry title.

param (
  [Parameter(Mandatory = $true)]
  [String]$Version,
  [String]$ServiceDirectory,
  [String]$PackageName,
  [Boolean]$Unreleased = $true,
  [Boolean]$ReplaceLatestEntryTitle = $false,
  [String]$ChangelogPath,
  [String]$ReleaseDate
)

. (Join-Path $PSScriptRoot common.ps1)

if ($ReleaseDate -and $Unreleased) {
    LogError "Do not pass 'ReleaseDate' arguement when 'Unreleased' is true"
    exit 1
}

if (!$PackageName -and !$ChangelogPath) {
    LogError "You must pass either the PackageName or ChangelogPath arguument."
    exit 1
}

if ($ReleaseDate)
{
    try {
        $ReleaseStatus = ([DateTime]$ReleaseDate).ToString($CHANGELOG_DATE_FORMAT)
        $ReleaseStatus = "($ReleaseStatus)"
    }
    catch {
        LogError "Invalid 'ReleaseDate'. Please use a valid date in the format '$CHANGELOG_DATE_FORMAT'. See https://aka.ms/azsdk/changelogguide"
        exit 1
    }
}
elseif ($Unreleased) 
{
    $ReleaseStatus = $CHANGELOG_UNRELEASED_STATUS
}
else 
{
    $ReleaseStatus = "$(Get-Date -Format $CHANGELOG_DATE_FORMAT)"
    $ReleaseStatus = "($ReleaseStatus)"
}

if ($null -eq [AzureEngSemanticVersion]::ParseVersionString($Version))
{
    LogError "Version [$Version] is invalid. Please use a valid SemVer. See https://aka.ms/azsdk/changelogguide"
    exit 1
}

if ([string]::IsNullOrEmpty($ChangelogPath))
{
    $pkgProperties = Get-PkgProperties -PackageName $PackageName -ServiceDirectory $ServiceDirectory
    $ChangelogPath = $pkgProperties.ChangeLogPath
}

if (!(Test-Path $ChangelogPath)) 
{
    LogError "Changelog path [$ChangelogPath] is invalid."
    exit 1
}

$ChangeLogEntries = Get-ChangeLogEntries -ChangeLogLocation $ChangelogPath

if ($ChangeLogEntries.Contains($Version))
{
    if ($ChangeLogEntries[$Version].ReleaseStatus -eq $ReleaseStatus)
    {
        LogWarning "Version [$Version] is already present in change log with specificed ReleaseStatus [$ReleaseStatus]. No Change made."
        exit(0)
    }

    if ($Unreleased -and ($ChangeLogEntries[$Version].ReleaseStatus -ne $ReleaseStatus))
    {
        LogWarning "Version [$Version] is already present in change log with a release date. Please review [$ChangelogPath]. No Change made."
        exit(0)
    }

    if (!$Unreleased -and ($ChangeLogEntries[$Version].ReleaseStatus -ne $CHANGELOG_UNRELEASED_STATUS))
    {
        if ((Get-Date ($ChangeLogEntries[$Version].ReleaseStatus).Trim("()")) -gt (Get-Date $ReleaseStatus.Trim("()")))
        {
            LogWarning "New ReleaseDate for version [$Version] is older than existing release date in changelog. Please review [$ChangelogPath]. No Change made."
            exit(0)
        }
    }
}

$PresentVersionsSorted = [AzureEngSemanticVersion]::SortVersionStrings($ChangeLogEntries.Keys)
$LatestVersion = $PresentVersionsSorted[0]

LogDebug "The latest release note entry in the changelog is for version [$($LatestVersion)]"

$LatestsSorted = [AzureEngSemanticVersion]::SortVersionStrings(@($LatestVersion, $Version))
if ($LatestsSorted[0] -ne $Version) {
    LogWarning "Version [$Version] is older than the latestversion [$LatestVersion] in the changelog. Consider using a more recent version."
}

if ($ReplaceLatestEntryTitle) 
{
    $newChangeLogEntry = New-ChangeLogEntry -Version $Version -Status $ReleaseStatus -Content $ChangeLogEntries[$LatestVersion].ReleaseContent
    LogDebug "Resetting latest entry title to [$($newChangeLogEntry.ReleaseTitle)]"
    $ChangeLogEntries.Remove($LatestVersion)
    if ($newChangeLogEntry) {
        $ChangeLogEntries.Insert(0, $Version, $newChangeLogEntry)
    }
    else {
        LogError "Failed to create new changelog entry"
        exit 1
    }
}
elseif ($ChangeLogEntries.Contains($Version))
{
    LogDebug "Updating ReleaseStatus for Version [$Version] to [$($ReleaseStatus)]"
    $ChangeLogEntries[$Version].ReleaseStatus = $ReleaseStatus
    $ChangeLogEntries[$Version].ReleaseTitle = "## $Version $ReleaseStatus"
}
else
{
    LogDebug "Adding new ChangeLog entry for Version [$Version]"
    $newChangeLogEntry = New-ChangeLogEntry -Version $Version -Status $ReleaseStatus
    if ($newChangeLogEntry) {
        $ChangeLogEntries.Insert(0, $Version, $newChangeLogEntry)
    }
    else {
        LogError "Failed to create new changelog entry"
        exit 1
    }
}

Set-ChangeLogContent -ChangeLogLocation $ChangelogPath -ChangeLogEntries $ChangeLogEntries