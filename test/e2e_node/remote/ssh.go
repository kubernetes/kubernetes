/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remote

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"strings"
	"sync"

	"k8s.io/klog/v2"
)

var sshOptions = flag.String("ssh-options", "", "Commandline options passed to ssh.")
var sshEnv = flag.String("ssh-env", "", "Use predefined ssh options for environment.  Options: gce")
var sshKey = flag.String("ssh-key", "", "Path to ssh private key.")
var sshUser = flag.String("ssh-user", "", "Use predefined user for ssh.")

var sshOptionsMap map[string]string
var sshDefaultKeyMap map[string]string
var sshDefaultUserMap map[string]string

func init() {
	usr, err := user.Current()
	if err != nil {
		klog.Fatal(err)
	}
	sshOptionsMap = map[string]string{
		"gce": "-o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o CheckHostIP=no -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o LogLevel=ERROR",
		"aws": "-o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o CheckHostIP=no -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o LogLevel=ERROR",
	}
	defaultGceKey := os.Getenv("GCE_SSH_PRIVATE_KEY_FILE")
	if defaultGceKey == "" {
		defaultGceKey = fmt.Sprintf("%s/.ssh/google_compute_engine", usr.HomeDir)
	}
	sshDefaultKeyMap = map[string]string{
		"gce": defaultGceKey,
	}
	sshDefaultUserMap = map[string]string{
		"aws": "ec2-user",
	}
}

var hostnameIPOverrides = struct {
	sync.RWMutex
	m map[string]string
}{m: make(map[string]string)}

// AddHostnameIP adds <hostname,ip> pair into hostnameIPOverrides map.
func AddHostnameIP(hostname, ip string) {
	hostnameIPOverrides.Lock()
	defer hostnameIPOverrides.Unlock()
	hostnameIPOverrides.m[hostname] = ip
}

var sshKeyOverrides = struct {
	sync.RWMutex
	m map[string]string
}{m: make(map[string]string)}

// AddSSHKey adds a <hosrtname,path to SSH private key> pair into the sshKeyOverrides map
func AddSSHKey(hostname, keyFilePath string) {
	sshKeyOverrides.Lock()
	defer sshKeyOverrides.Unlock()
	sshKeyOverrides.m[hostname] = keyFilePath
}

// GetSSHUser returns the ssh-user CLI flag, the KUBE_SSH_USER environment variable, or the default ssh user
// for the ssh environment in that order
func GetSSHUser() string {
	if *sshUser == "" {
		*sshUser = os.Getenv("KUBE_SSH_USER")
	}
	if *sshUser == "" {
		*sshUser = sshDefaultUserMap[*sshEnv]
	}
	return *sshUser
}

// GetHostnameOrIP converts hostname into ip and apply user if necessary.
func GetHostnameOrIP(hostname string) string {
	hostnameIPOverrides.RLock()
	defer hostnameIPOverrides.RUnlock()
	host := hostname
	if ip, found := hostnameIPOverrides.m[hostname]; found {
		host = ip
	}

	sshUser := GetSSHUser()
	if sshUser != "" {
		host = fmt.Sprintf("%s@%s", sshUser, host)
	}
	return host
}

// getSSHCommand handles proper quoting so that multiple commands are executed in the same shell over ssh
func getSSHCommand(sep string, args ...string) string {
	return fmt.Sprintf("'%s'", strings.Join(args, sep))
}

// SSH executes ssh command with runSSHCommand as root. The `sudo` makes sure that all commands
// are executed by root, so that there won't be permission mismatch between different commands.
func SSH(host string, cmd ...string) (string, error) {
	return runSSHCommand(host, "ssh", append([]string{GetHostnameOrIP(host), "--", "sudo"}, cmd...)...)
}

// SSHNoSudo executes ssh command with runSSHCommand as normal user. Sometimes we need this,
// for example creating a directory that we'll copy files there with scp.
func SSHNoSudo(host string, cmd ...string) (string, error) {
	return runSSHCommand(host, "ssh", append([]string{GetHostnameOrIP(host), "--"}, cmd...)...)
}

// runSSHCommand executes the ssh or scp command, adding the flag provided --ssh-options
func runSSHCommand(host, cmd string, args ...string) (string, error) {
	if key, err := getPrivateSSHKey(host); len(key) != 0 {
		if err != nil {
			klog.Errorf("private SSH key (%s) not found. Check if the SSH key is configured properly:, err: %v", key, err)
			return "", fmt.Errorf("private SSH key (%s) does not exist", key)
		}

		args = append([]string{"-i", key}, args...)
	}
	if env, found := sshOptionsMap[*sshEnv]; found {
		args = append(strings.Split(env, " "), args...)
	}
	if *sshOptions != "" {
		args = append(strings.Split(*sshOptions, " "), args...)
	}
	klog.Infof("Running the command %s, with args: %v", cmd, args)
	output, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		klog.Errorf("failed to run SSH command: out: %s, err: %v", output, err)
		return string(output), fmt.Errorf("command [%s %s] failed with error: %w", cmd, strings.Join(args, " "), err)
	}
	return string(output), nil
}

// getPrivateSSHKey returns the path to ssh private key
func getPrivateSSHKey(host string) (string, error) {
	if *sshKey != "" {
		if _, err := os.Stat(*sshKey); err != nil {
			return *sshKey, err
		}

		return *sshKey, nil
	}

	sshKeyOverrides.Lock()
	defer sshKeyOverrides.Unlock()
	if key, ok := sshKeyOverrides.m[host]; ok {
		return key, nil
	}

	if key, found := sshDefaultKeyMap[*sshEnv]; found {
		if _, err := os.Stat(key); err != nil {
			return key, err
		}

		return key, nil
	}

	return "", nil
}
