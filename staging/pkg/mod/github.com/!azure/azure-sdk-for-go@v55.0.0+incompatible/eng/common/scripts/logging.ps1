$isDevOpsRun = ($null -ne $env:SYSTEM_TEAMPROJECTID)

function LogWarning
{
    if ($isDevOpsRun) 
    {
        Write-Host "##vso[task.LogIssue type=warning;]$args"
    }
    else
    {
        Write-Warning "$args"
    }
}

function LogError
{
    if ($isDevOpsRun) 
    {
        Write-Host "##vso[task.LogIssue type=error;]$args"
    }
    else 
    {
        Write-Error "$args"
    }
}

function LogDebug
{
    if ($isDevOpsRun) 
    {
        Write-Host "[debug]$args"
    }
    else 
    {
        Write-Debug "$args"
    }
}
