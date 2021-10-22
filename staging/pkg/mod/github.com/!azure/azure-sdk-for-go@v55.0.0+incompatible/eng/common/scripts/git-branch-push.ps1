 #!/usr/bin/env pwsh -c

<#
.DESCRIPTION
Create local branch of the given repo and attempt to push changes. The push may fail if
there has been other changes pushed to the same branch, if so, fetch, rebase and try again.
.PARAMETER PRBranchName
The name of the github branch the changes are being put into
.PARAMETER CommitMsg
The message for this particular commit
.PARAMETER GitUrl
The GitHub repository URL
.PARAMETER PushArgs
Optional arguments to the push command
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string] $PRBranchName,

    [Parameter(Mandatory = $true)]
    [string] $CommitMsg,

    [Parameter(Mandatory = $true)]
    [string] $GitUrl,

    [Parameter(Mandatory = $false)]
    [string] $PushArgs = "",

    [Parameter(Mandatory = $false)]
    [string] $RemoteName = "azure-sdk-fork",

    [Parameter(Mandatory = $false)]
    [boolean] $SkipCommit = $false,

    [Parameter(Mandatory = $false)]
    [boolean] $AmendCommit = $false
)

# This is necessay because of the janky git command output writing to stderr.
# Without explicitly setting the ErrorActionPreference to continue the script
# would fail the first time git wrote command output.
$ErrorActionPreference = "Continue"

if ((git remote) -contains $RemoteName)
{
  Write-Host "git remote get-url $RemoteName"
  $remoteUrl = git remote get-url $RemoteName
  if ($remoteUrl -ne $GitUrl)
  {
    Write-Error "Remote with name $RemoteName already exists with an incompatible url [$remoteUrl] which should be [$GitUrl]."
    exit 1
  }
}
else 
{
  Write-Host "git remote add $RemoteName $GitUrl"
  git remote add $RemoteName $GitUrl
  if ($LASTEXITCODE -ne 0)
  {
    Write-Error "Unable to add remote LASTEXITCODE=$($LASTEXITCODE), see command output above."
    exit $LASTEXITCODE
  }
}
# Checkout to $PRBranch, create new one if not exists.
git show-ref --verify --quiet refs/heads/$PRBranchName
if ($LASTEXITCODE -eq 0) {
  Write-Host "git checkout $PRBranchName."
  git checkout $PRBranchName 
} 
else {
  Write-Host "git checkout -b $PRBranchName."
  git checkout -b $PRBranchName
}
if ($LASTEXITCODE -ne 0)
{
    Write-Error "Unable to create branch LASTEXITCODE=$($LASTEXITCODE), see command output above."
    exit $LASTEXITCODE
}

if (!$SkipCommit) {
    if ($AmendCommit) {
        $amendOption = "--amend"
    }
    else {
        $amendOption = ""
    }
    Write-Host "git -c user.name=`"azure-sdk`" -c user.email=`"azuresdk@microsoft.com`" commit $amendOption -am `"$($CommitMsg)`""
    git -c user.name="azure-sdk" -c user.email="azuresdk@microsoft.com" commit $amendOption -am "$($CommitMsg)"
    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Unable to add files and create commit LASTEXITCODE=$($LASTEXITCODE), see command output above."
        exit $LASTEXITCODE
    }
}
else {
    Write-Host "Skipped applying commit"
}

# The number of retries can be increased if necessary. In theory, the number of retries
# should be the max number of libraries in the largest pipeline -1 as everything except
# the first commit could hit issues and need to rebase. The reason this isn't set to that
# is because the largest pipeline is cognitive services which has 18 libraries in its
# pipeline and that just seemed a bit too large and 10 seemed like a good starting value.
$numberOfRetries = 10
$needsRetry = $false
$tryNumber = 0
do
{
    $needsRetry = $false
    Write-Host "git push $RemoteName $PRBranchName $PushArgs"
    git push $RemoteName $PRBranchName $PushArgs
    $tryNumber++
    if ($LASTEXITCODE -ne 0)
    {
        $needsRetry = $true
        Write-Host "Git push failed with LASTEXITCODE=$($LASTEXITCODE) Need to fetch and rebase: attempt number=$($tryNumber)"
 
        Write-Host "git fetch $RemoteName"
        git fetch $RemoteName
        if ($LASTEXITCODE -ne 0)
        {
            Write-Error "Unable to fetch remote LASTEXITCODE=$($LASTEXITCODE), see command output above."
            exit $LASTEXITCODE
        }

        try
        {
            $TempPatchFile = New-TemporaryFile
            Write-Host "git diff ${PRBranchName}~ ${PRBranchName} --output $TempPatchFile"
            git diff ${PRBranchName}~ ${PRBranchName} --output $TempPatchFile
            if ($LASTEXITCODE -ne 0)
            {
                Write-Error "Unable to create diff file LASTEXITCODE=$($LASTEXITCODE), see command output above."
                continue
            }

            Write-Host "git reset --hard $RemoteName/${PRBranchName}"
            git reset --hard $RemoteName/${PRBranchName}
            if ($LASTEXITCODE -ne 0)
            {
                Write-Error "Unable to hard reset branch LASTEXITCODE=$($LASTEXITCODE), see command output above."
                continue
            }

            # -C0 means to use no extra before or after lines of context to enable us to avoid adjacent line merge conflicts
            Write-Host "git apply -C0 $TempPatchFile"
            git apply -C0 $TempPatchFile
            if ($LASTEXITCODE -ne 0)
            {
                Write-Error "Unable to apply diff file LASTEXITCODE=$($LASTEXITCODE), see command output above."
                exit $LASTEXITCODE
            }


            Write-Host "git add -A"
            git add -A
            if ($LASTEXITCODE -ne 0)
            {
                Write-Error "Unable to git add LASTEXITCODE=$($LASTEXITCODE), see command output above."
                continue
            }

            Write-Host "git -c user.name=`"azure-sdk`" -c user.email=`"azuresdk@microsoft.com`" commit -m `"$($CommitMsg)`""
            git -c user.name="azure-sdk" -c user.email="azuresdk@microsoft.com" commit -m "$($CommitMsg)"
            if ($LASTEXITCODE -ne 0)
            {
                Write-Error "Unable to commit LASTEXITCODE=$($LASTEXITCODE), see command output above."
                continue
            }
        }
        finally
        {
            if ( Test-Path $TempPatchFile )
            {
                Remove-Item $TempPatchFile
            }
        }
    }
} while($needsRetry -and $tryNumber -le $numberOfRetries)

if ($LASTEXITCODE -ne 0 -or $tryNumber -gt $numberOfRetries)
{
    Write-Error "Unable to push commit after $($tryNumber) retries LASTEXITCODE=$($LASTEXITCODE), see command output above."
    if (0 -eq $LASTEXITCODE) 
    {
        exit 1
    }
    exit $LASTEXITCODE
}
