Param(
  [string] $serviceDir = ""
)

if($serviceDir){
  $targetDir = "$PSScriptRoot/../../sdk/$serviceDir"
}
else {
  $targetDir = "$PSScriptRoot/../../sdk"
}

$path = Resolve-Path -Path $targetDir

return "$path/..."
