package kubeadm

import "testing"

// kubeadmReset executes "kubeadm reset" and restarts kubelet.
func kubeadmReset() error {
	_, _, err := RunCmd(kubeadmPath, "reset")
	return err
}

func TestCmdInitToken(t *testing.T) {
	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--discovery=token://abcd:1234567890abcd", false},     // invalid token size
		{"--discovery=token://Abcdef:1234567890abcdef", false}, // invalid token non-lowercase
		{"--discovery=token://abcdef:1234567890abcdef", true},  // valid token
		{"", true}, // no token provided, so generate
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(kubeadmPath, "init", rt.args, "--skip-preflight-checks")
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdInitToken running 'kubeadm init %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}
		kubeadmReset()
	}
}
