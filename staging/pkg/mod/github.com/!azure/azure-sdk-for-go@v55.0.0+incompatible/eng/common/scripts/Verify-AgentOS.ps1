param (
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $AgentImage
)

function Throw-InvalidOperatingSystem {
    throw "Invalid operating system detected. Operating system was: $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription), expected image was: $AgentImage"
}

if ($IsWindows -and $AgentImage -match "windows|win|MMS2019") {
    $osName = "Windows"
} elseif ($IsLinux -and $AgentImage -match "ubuntu") {
    $osName = "Linux"
} elseif ($IsMacOs -and $AgentImage -match "macos") {
    $osName = "macOS"
} else {
    Throw-InvalidOperatingSystem
}

Write-Host "##vso[task.setvariable variable=OSName]$osName"
