# Copyright 2019 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Param(
  [string]$FileName = $(throw "-FileName is required.")
 )


# read = read data | read attributes
$READ_PERMISSIONS = 0x0001 -bor 0x0080

# write = write data | append data | write attributes | write EA
$WRITE_PERMISSIONS = 0x0002 -bor 0x0004 -bor 0x0100 -bor  0x0010

# execute = read data | file execute
$EXECUTE_PERMISSIONS = 0x0001 -bor 0x0020


function GetFilePermissions($path) {
    $objPath = "Win32_LogicalFileSecuritySetting='$path'"
    $output = Invoke-WmiMethod -Namespace root/cimv2 -Path $objPath -Name GetSecurityDescriptor

    if ($output.ReturnValue -ne 0) {
        $retVal = $output.ReturnValue
        Write-Error "GetSecurityDescriptor invocation failed with code: $retVal"
        exit 1
    }

    $fileSD = $output.Descriptor
    $fileOwnerGroup = $fileSD.Group
    $fileOwner = $fileSD.Owner

    if ($fileOwnerGroup.Name -eq $null -and $fileOwnerGroup.Domain -eq $null) {
        # the file owner's group is not recognized. Check if the Owner itself is
        # a group, and if so, default the group to it.
        net user $fileOwner.Name > $null 2> $null
        if (-not $?) {
            $fileOwnerGroup = $fileOwner
        }

    }

    $userMask = 0
    $groupMask = 0
    $otherMask = 0

    foreach ($ace in $fileSD.DACL) {
        $mask = 0
        if ($ace.AceType -ne 0) {
            # not an Allow ACE, skip.
            continue
        }

        # convert mask.
        if ( ($ace.AccessMask -band $READ_PERMISSIONS) -eq $READ_PERMISSIONS ) {
            $mask = $mask -bor 4
        }
        if ( ($ace.AccessMask -band $WRITE_PERMISSIONS) -eq $WRITE_PERMISSIONS ) {
            $mask = $mask -bor 2
        }
        if ( ($ace.AccessMask -band $EXECUTE_PERMISSIONS) -eq $EXECUTE_PERMISSIONS ) {
            $mask = $mask -bor 1
        }

        # detect mask type.
        if ($ace.Trustee.Equals($fileOwner)) {
            $userMask = $mask
        }
        if ($ace.Trustee.Equals($fileOwnerGroup)) {
            $groupMask = $mask
        }
        if ($ace.Trustee.Name.ToLower() -eq "users") {
            $otherMask = $mask
        }
    }

    return "$userMask$groupMask$otherMask"
}

$mask = GetFilePermissions($FileName)
if (-not $?) {
    exit 1
}

# print the permission mask Linux-style.
echo "0$mask"
