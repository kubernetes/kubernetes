// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package e2e

import (
	"fmt"
	"testing"
)

func TestCtlV3RoleAdd(t *testing.T)          { testCtl(t, roleAddTest) }
func TestCtlV3RoleAddNoTLS(t *testing.T)     { testCtl(t, roleAddTest, withCfg(configNoTLS)) }
func TestCtlV3RoleAddClientTLS(t *testing.T) { testCtl(t, roleAddTest, withCfg(configClientTLS)) }
func TestCtlV3RoleAddPeerTLS(t *testing.T)   { testCtl(t, roleAddTest, withCfg(configPeerTLS)) }
func TestCtlV3RoleAddTimeout(t *testing.T)   { testCtl(t, roleAddTest, withDialTimeout(0)) }

func TestCtlV3RoleGrant(t *testing.T) { testCtl(t, roleGrantTest) }

func roleAddTest(cx ctlCtx) {
	cmdSet := []struct {
		args        []string
		expectedStr string
	}{
		// Add a role.
		{
			args:        []string{"add", "root"},
			expectedStr: "Role root created",
		},
		// Try adding the same role.
		{
			args:        []string{"add", "root"},
			expectedStr: "role name already exists",
		},
	}

	for i, cmd := range cmdSet {
		if err := ctlV3Role(cx, cmd.args, cmd.expectedStr); err != nil {
			if cx.dialTimeout > 0 && !isGRPCTimedout(err) {
				cx.t.Fatalf("roleAddTest #%d: ctlV3Role error (%v)", i, err)
			}
		}
	}
}

func roleGrantTest(cx ctlCtx) {
	cmdSet := []struct {
		args        []string
		expectedStr string
	}{
		// Add a role.
		{
			args:        []string{"add", "root"},
			expectedStr: "Role root created",
		},
		// Grant read permission to the role.
		{
			args:        []string{"grant", "root", "read", "foo"},
			expectedStr: "Role root updated",
		},
		// Grant write permission to the role.
		{
			args:        []string{"grant", "root", "write", "foo"},
			expectedStr: "Role root updated",
		},
		// Grant rw permission to the role.
		{
			args:        []string{"grant", "root", "readwrite", "foo"},
			expectedStr: "Role root updated",
		},
		// Try granting invalid permission to the role.
		{
			args:        []string{"grant", "root", "123", "foo"},
			expectedStr: "invalid permission type",
		},
	}

	for i, cmd := range cmdSet {
		if err := ctlV3Role(cx, cmd.args, cmd.expectedStr); err != nil {
			cx.t.Fatalf("roleGrantTest #%d: ctlV3Role error (%v)", i, err)
		}
	}
}

func ctlV3Role(cx ctlCtx, args []string, expStr string) error {
	cmdArgs := append(cx.PrefixArgs(), "role")
	cmdArgs = append(cmdArgs, args...)

	return spawnWithExpect(cmdArgs, expStr)
}

func ctlV3RoleGrantPermission(cx ctlCtx, rolename string, perm grantingPerm) error {
	cmdArgs := append(cx.PrefixArgs(), "role", "grant-permission")
	if perm.prefix {
		cmdArgs = append(cmdArgs, "--prefix")
	}
	cmdArgs = append(cmdArgs, rolename)
	cmdArgs = append(cmdArgs, grantingPermToArgs(perm)...)

	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return err
	}

	expStr := fmt.Sprintf("Role %s updated", rolename)
	_, err = proc.Expect(expStr)
	return err
}

func ctlV3RoleRevokePermission(cx ctlCtx, rolename string, key, rangeEnd string) error {
	cmdArgs := append(cx.PrefixArgs(), "role", "revoke-permission")
	cmdArgs = append(cmdArgs, rolename)
	cmdArgs = append(cmdArgs, key)
	if len(rangeEnd) != 0 {
		cmdArgs = append(cmdArgs, rangeEnd)
	}

	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return err
	}

	expStr := fmt.Sprintf("Permission of key %s is revoked from role %s", key, rolename)
	_, err = proc.Expect(expStr)
	return err
}

type grantingPerm struct {
	read     bool
	write    bool
	key      string
	rangeEnd string
	prefix   bool
}

func grantingPermToArgs(perm grantingPerm) []string {
	permstr := ""

	if perm.read {
		permstr += "read"
	}

	if perm.write {
		permstr += "write"
	}

	if len(permstr) == 0 {
		panic("invalid granting permission")
	}

	if len(perm.rangeEnd) == 0 {
		return []string{permstr, perm.key}
	}
	return []string{permstr, perm.key, perm.rangeEnd}
}
