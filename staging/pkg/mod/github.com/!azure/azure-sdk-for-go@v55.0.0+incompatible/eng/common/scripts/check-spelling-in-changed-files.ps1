[CmdletBinding()]
Param (
    [Parameter()]
    [string] $TargetBranch,

    [Parameter()]
    [string] $SourceBranch,

    [Parameter()]
    [string] $CspellConfigPath = "./.vscode/cspell.json"
)

$ErrorActionPreference = "Continue"
. $PSScriptRoot/logging.ps1

if ((Get-Command git | Measure-Object).Count -eq 0) { 
    LogError "Could not locate git. Install git https://git-scm.com/downloads"
    exit 1
}

if ((Get-Command npx | Measure-Object).Count -eq 0) { 
    LogError "Could not locate npx. Install NodeJS (includes npm and npx) https://nodejs.org/en/download/"
    exit 1
}

if (!(Test-Path $CspellConfigPath)) {
    LogError "Could not locate config file $CspellConfigPath"
    exit 1
}

# Lists names of files that were in some way changed between the 
# current $SourceBranch and $TargetBranch. Excludes files that were deleted to
# prevent errors in Resolve-Path
Write-Host "git diff --diff-filter=d --name-only $TargetBranch $SourceBranch"
$changedFiles = git diff --diff-filter=d --name-only $TargetBranch $SourceBranch `
    | Resolve-Path

$changedFilesCount = ($changedFiles | Measure-Object).Count
Write-Host "Git Detected $changedFilesCount changed file(s). Files checked by cspell may exclude files according to cspell.json"

if ($changedFilesCount -eq 0) {
    Write-Host "No changes detected"
    exit 0
}

$changedFilesString = $changedFiles -join ' '

Write-Host "npx cspell --config $CspellConfigPath $changedFilesString"
$spellingErrors = Invoke-Expression "npx cspell --config $CspellConfigPath $changedFilesString"

if ($spellingErrors) {
    foreach ($spellingError in $spellingErrors) { 
        LogWarning $spellingError
    }
    LogWarning "Spelling errors detected. To correct false positives or learn about spell checking see: https://aka.ms/azsdk/engsys/spellcheck"
}

exit 0

<#
.SYNOPSIS
Uses cspell (from NPM) to check spelling of recently changed files

.DESCRIPTION
This script checks files that have changed relative to a base branch (default 
branch) for spelling errors. Dictionaries and spelling configurations reside 
in a configurable `cspell.json` location.

This script uses `npx` and assumes that NodeJS (and by extension `npm` and `npx`
) are installed on the machine. If it does not detect `npx` it will warn the 
user and exit with an error. 

The entire file is scanned, not just changed sections. Spelling errors in parts 
of the file not touched will still be shown.

Running this on the local machine will trigger tests 

.PARAMETER TargetBranch
Git ref to compare changes. This is usually the "base" (GitHub) or "target" 
(DevOps) branch for which a pull request would be opened.

.PARAMETER SouceBranch
Git ref to use instead of changes in current repo state. Use `HEAD` here to 
check spelling of files that have been committed and exlcude any new files or
modified files that are not committed. This is most useful in CI scenarios where
builds may have modified the state of the repo. Leaving this parameter blank  
includes files for whom changes have not been committed. 

.PARAMETER CspellConfigPath
Optional location to use for cspell.json path. Default value is 
`./.vscode/cspell.json`

.EXAMPLE
./eng/common/scripts/check-spelling-in-changed-files.ps1 -TargetBranch 'target_branch_name'

This will run spell check with changes in the current branch with respect to 
`target_branch_name`

#>
