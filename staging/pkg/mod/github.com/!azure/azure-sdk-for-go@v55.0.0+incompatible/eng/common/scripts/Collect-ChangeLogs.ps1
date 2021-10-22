[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [DateTime] $FromDate
)

. (Join-Path $PSScriptRoot common.ps1)

$releaseHighlights = @{}

if ($FromDate -as [DateTime])
{
    $date = ([DateTime]$FromDate).ToString($CHANGELOG_DATE_FORMAT)
}
else {
    LogWarning "Invalid date passed. Switch to using the current date"
    $date = Get-Date -Format $CHANGELOG_DATE_FORMAT
}

$allPackageProps = Get-AllPkgProperties

foreach ($packageProp in $allPackageProps) {
    $changeLogLocation = $packageProp.ChangeLogPath
    if (!(Test-Path $changeLogLocation))
    {
        continue
    }
    $changeLogEntries = Get-ChangeLogEntries -ChangeLogLocation $changeLogLocation
    $packageName = $packageProp.Name
    $serviceDirectory = $packageProp.ServiceDirectory
    $packageDirectoryname = Split-Path -Path $packageProp.DirectoryPath -Leaf

    foreach ($changeLogEntry in $changeLogEntries.Values) {
        if ([System.String]::IsNullOrEmpty($changeLogEntry.ReleaseStatus))
        {
            continue;
        }
        $ReleaseStatus = $changeLogEntry.ReleaseStatus.Trim("(",")")
        if (!($ReleaseStatus -as [DateTime]) -or $ReleaseStatus -lt $date)
        {
            continue;
        }

        $releaseVersion = $changeLogEntry.ReleaseVersion
        $githubAnchor = $changeLogEntry.ReleaseTitle.Replace("## ", "").Replace(".", "").Replace("(", "").Replace(")", "").Replace(" ", "-")

        $releaseTag = "${packageName}_${releaseVersion}"
        $key = "${packageName}:${releaseVersion}"

        $releaseHighlights[$key] = @{}
        $releaseHighlights[$key]["PackageProperties"] = $packageProp
        $releaseHighlights[$key]["ChangelogUrl"] = "https://github.com/Azure/azure-sdk-for-${LanguageShort}/blob/${releaseTag}/sdk/${serviceDirectory}/${packageDirectoryname}/CHANGELOG.md#${githubAnchor}"
        $releaseHighlights[$key]["Content"] = @()

        $changeLogEntry.ReleaseContent | %{
            $releaseHighlights[$key]["Content"] += $_.Replace("###", "####")
        }
    }
}

return $releaseHighlights