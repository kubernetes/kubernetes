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
    $fileAcl = Get-Acl -Path $path
    $fileOwner = $fileAcl.Owner
    $fileGroup = $fileAcl.Group

    $userMask = 0
    $groupMask = 0
    $otherMask = 0

    foreach ($rule in $fileAcl.Access) {
        if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) {
            # not an allow rule, skipping.
            continue
        }

        $mask = 0
        $rights = $rule.FileSystemRights.value__
        # convert mask.
        if ( ($rights -band $READ_PERMISSIONS) -eq $READ_PERMISSIONS ) {
            $mask = $mask -bor 4
        }
        if ( ($rights -band $WRITE_PERMISSIONS) -eq $WRITE_PERMISSIONS ) {
            $mask = $mask -bor 2
        }
        if ( ($rights -band $EXECUTE_PERMISSIONS) -eq $EXECUTE_PERMISSIONS ) {
            $mask = $mask -bor 1
        }

        # detect mask type.
        if ($rule.IdentityReference.Value.Equals($fileOwner)) {
            $userMask = $mask
        }
        if ($rule.IdentityReference.Value.Equals($fileGroup)) {
            $groupMask = $mask
        }
        if ($rule.IdentityReference.Value.ToLower().Contains("users")) {
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
