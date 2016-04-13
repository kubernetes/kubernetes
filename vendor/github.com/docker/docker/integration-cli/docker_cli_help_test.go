package main

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
	"unicode"

	"github.com/docker/docker/pkg/homedir"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestHelpTextVerify(c *check.C) {
	// Make sure main help text fits within 80 chars and that
	// on non-windows system we use ~ when possible (to shorten things).
	// Test for HOME set to its default value and set to "/" on linux
	// Yes on windows setting up an array and looping (right now) isn't
	// necessary because we just have one value, but we'll need the
	// array/loop on linux so we might as well set it up so that we can
	// test any number of home dirs later on and all we need to do is
	// modify the array - the rest of the testing infrastructure should work
	homes := []string{homedir.Get()}

	// Non-Windows machines need to test for this special case of $HOME
	if runtime.GOOS != "windows" {
		homes = append(homes, "/")
	}

	homeKey := homedir.Key()
	baseEnvs := os.Environ()

	// Remove HOME env var from list so we can add a new value later.
	for i, env := range baseEnvs {
		if strings.HasPrefix(env, homeKey+"=") {
			baseEnvs = append(baseEnvs[:i], baseEnvs[i+1:]...)
			break
		}
	}

	for _, home := range homes {
		// Dup baseEnvs and add our new HOME value
		newEnvs := make([]string, len(baseEnvs)+1)
		copy(newEnvs, baseEnvs)
		newEnvs[len(newEnvs)-1] = homeKey + "=" + home

		scanForHome := runtime.GOOS != "windows" && home != "/"

		// Check main help text to make sure its not over 80 chars
		helpCmd := exec.Command(dockerBinary, "help")
		helpCmd.Env = newEnvs
		out, ec, err := runCommandWithOutput(helpCmd)
		if err != nil || ec != 0 {
			c.Fatalf("docker help should have worked\nout:%s\nec:%d", out, ec)
		}
		lines := strings.Split(out, "\n")
		for _, line := range lines {
			if len(line) > 80 {
				c.Fatalf("Line is too long(%d chars):\n%s", len(line), line)
			}

			// All lines should not end with a space
			if strings.HasSuffix(line, " ") {
				c.Fatalf("Line should not end with a space: %s", line)
			}

			if scanForHome && strings.Contains(line, `=`+home) {
				c.Fatalf("Line should use '%q' instead of %q:\n%s", homedir.GetShortcutString(), home, line)
			}
			if runtime.GOOS != "windows" {
				i := strings.Index(line, homedir.GetShortcutString())
				if i >= 0 && i != len(line)-1 && line[i+1] != '/' {
					c.Fatalf("Main help should not have used home shortcut:\n%s", line)
				}
			}
		}

		// Make sure each cmd's help text fits within 80 chars and that
		// on non-windows system we use ~ when possible (to shorten things).
		// Pull the list of commands from the "Commands:" section of docker help
		helpCmd = exec.Command(dockerBinary, "help")
		helpCmd.Env = newEnvs
		out, ec, err = runCommandWithOutput(helpCmd)
		if err != nil || ec != 0 {
			c.Fatalf("docker help should have worked\nout:%s\nec:%d", out, ec)
		}
		i := strings.Index(out, "Commands:")
		if i < 0 {
			c.Fatalf("Missing 'Commands:' in:\n%s", out)
		}

		// Grab all chars starting at "Commands:"
		// Skip first line, its "Commands:"
		cmds := []string{}
		for _, cmd := range strings.Split(out[i:], "\n")[1:] {
			var stderr string

			// Stop on blank line or non-idented line
			if cmd == "" || !unicode.IsSpace(rune(cmd[0])) {
				break
			}

			// Grab just the first word of each line
			cmd = strings.Split(strings.TrimSpace(cmd), " ")[0]
			cmds = append(cmds, cmd)

			// Check the full usage text
			helpCmd := exec.Command(dockerBinary, cmd, "--help")
			helpCmd.Env = newEnvs
			out, stderr, ec, err = runCommandWithStdoutStderr(helpCmd)
			if len(stderr) != 0 {
				c.Fatalf("Error on %q help. non-empty stderr:%q", cmd, stderr)
			}
			if strings.HasSuffix(out, "\n\n") {
				c.Fatalf("Should not have blank line on %q\nout:%q", cmd, out)
			}
			if !strings.Contains(out, "--help=false") {
				c.Fatalf("Should show full usage on %q\nout:%q", cmd, out)
			}
			if err != nil || ec != 0 {
				c.Fatalf("Error on %q help: %s\nexit code:%d", cmd, out, ec)
			}

			// Check each line for lots of stuff
			lines := strings.Split(out, "\n")
			for _, line := range lines {
				if len(line) > 80 {
					c.Fatalf("Help for %q is too long(%d chars):\n%s", cmd,
						len(line), line)
				}

				if scanForHome && strings.Contains(line, `"`+home) {
					c.Fatalf("Help for %q should use ~ instead of %q on:\n%s",
						cmd, home, line)
				}
				i := strings.Index(line, "~")
				if i >= 0 && i != len(line)-1 && line[i+1] != '/' {
					c.Fatalf("Help for %q should not have used ~:\n%s", cmd, line)
				}

				// If a line starts with 4 spaces then assume someone
				// added a multi-line description for an option and we need
				// to flag it
				if strings.HasPrefix(line, "    ") {
					c.Fatalf("Help for %q should not have a multi-line option: %s", cmd, line)
				}

				// Options should NOT end with a period
				if strings.HasPrefix(line, "  -") && strings.HasSuffix(line, ".") {
					c.Fatalf("Help for %q should not end with a period: %s", cmd, line)
				}

				// Options should NOT end with a space
				if strings.HasSuffix(line, " ") {
					c.Fatalf("Help for %q should not end with a space: %s", cmd, line)
				}

			}

			// For each command make sure we generate an error
			// if we give a bad arg
			dCmd := exec.Command(dockerBinary, cmd, "--badArg")
			out, stderr, ec, err = runCommandWithStdoutStderr(dCmd)
			if len(out) != 0 || len(stderr) == 0 || ec == 0 || err == nil {
				c.Fatalf("Bad results from 'docker %s --badArg'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", cmd, ec, out, stderr, err)
			}
			// Be really picky
			if strings.HasSuffix(stderr, "\n\n") {
				c.Fatalf("Should not have a blank line at the end of 'docker rm'\n%s", stderr)
			}

			// Now make sure that each command will print a short-usage
			// (not a full usage - meaning no opts section) if we
			// are missing a required arg or pass in a bad arg

			// These commands will never print a short-usage so don't test
			noShortUsage := map[string]string{
				"images": "",
				"login":  "",
				"logout": "",
			}

			if _, ok := noShortUsage[cmd]; !ok {
				// For each command run it w/o any args. It will either return
				// valid output or print a short-usage
				var dCmd *exec.Cmd
				var stdout, stderr string
				var args []string

				// skipNoArgs are ones that we don't want to try w/o
				// any args. Either because it'll hang the test or
				// lead to incorrect test result (like false negative).
				// Whatever the reason, skip trying to run w/o args and
				// jump to trying with a bogus arg.
				skipNoArgs := map[string]string{
					"events": "",
					"load":   "",
				}

				ec = 0
				if _, ok := skipNoArgs[cmd]; !ok {
					args = []string{cmd}
					dCmd = exec.Command(dockerBinary, args...)
					stdout, stderr, ec, err = runCommandWithStdoutStderr(dCmd)
				}

				// If its ok w/o any args then try again with an arg
				if ec == 0 {
					args = []string{cmd, "badArg"}
					dCmd = exec.Command(dockerBinary, args...)
					stdout, stderr, ec, err = runCommandWithStdoutStderr(dCmd)
				}

				if len(stdout) != 0 || len(stderr) == 0 || ec == 0 || err == nil {
					c.Fatalf("Bad output from %q\nstdout:%q\nstderr:%q\nec:%d\nerr:%q", args, stdout, stderr, ec, err)
				}
				// Should have just short usage
				if !strings.Contains(stderr, "\nUsage:\t") {
					c.Fatalf("Missing short usage on %q\nstderr:%q", args, stderr)
				}
				// But shouldn't have full usage
				if strings.Contains(stderr, "--help=false") {
					c.Fatalf("Should not have full usage on %q\nstderr:%q", args, stderr)
				}
				if strings.HasSuffix(stderr, "\n\n") {
					c.Fatalf("Should not have a blank line on %q\nstderr:%q", args, stderr)
				}
			}

		}

		expected := 39
		if len(cmds) != expected {
			c.Fatalf("Wrong # of cmds(%d), it should be: %d\nThe list:\n%q",
				len(cmds), expected, cmds)
		}
	}

}

func (s *DockerSuite) TestHelpExitCodesHelpOutput(c *check.C) {
	// Test to make sure the exit code and output (stdout vs stderr) of
	// various good and bad cases are what we expect

	// docker : stdout=all, stderr=empty, rc=0
	cmd := exec.Command(dockerBinary)
	stdout, stderr, ec, err := runCommandWithStdoutStderr(cmd)
	if len(stdout) == 0 || len(stderr) != 0 || ec != 0 || err != nil {
		c.Fatalf("Bad results from 'docker'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Be really pick
	if strings.HasSuffix(stdout, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker'\n%s", stdout)
	}

	// docker help: stdout=all, stderr=empty, rc=0
	cmd = exec.Command(dockerBinary, "help")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) == 0 || len(stderr) != 0 || ec != 0 || err != nil {
		c.Fatalf("Bad results from 'docker help'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Be really pick
	if strings.HasSuffix(stdout, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker help'\n%s", stdout)
	}

	// docker --help: stdout=all, stderr=empty, rc=0
	cmd = exec.Command(dockerBinary, "--help")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) == 0 || len(stderr) != 0 || ec != 0 || err != nil {
		c.Fatalf("Bad results from 'docker --help'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Be really pick
	if strings.HasSuffix(stdout, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker --help'\n%s", stdout)
	}

	// docker inspect busybox: stdout=all, stderr=empty, rc=0
	// Just making sure stderr is empty on valid cmd
	cmd = exec.Command(dockerBinary, "inspect", "busybox")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) == 0 || len(stderr) != 0 || ec != 0 || err != nil {
		c.Fatalf("Bad results from 'docker inspect busybox'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Be really pick
	if strings.HasSuffix(stdout, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker inspect busyBox'\n%s", stdout)
	}

	// docker rm: stdout=empty, stderr=all, rc!=0
	// testing the min arg error msg
	cmd = exec.Command(dockerBinary, "rm")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) != 0 || len(stderr) == 0 || ec == 0 || err == nil {
		c.Fatalf("Bad results from 'docker rm'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Should not contain full help text but should contain info about
	// # of args and Usage line
	if !strings.Contains(stderr, "requires a minimum") {
		c.Fatalf("Missing # of args text from 'docker rm'\nstderr:%s", stderr)
	}

	// docker rm NoSuchContainer: stdout=empty, stderr=all, rc=0
	// testing to make sure no blank line on error
	cmd = exec.Command(dockerBinary, "rm", "NoSuchContainer")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) != 0 || len(stderr) == 0 || ec == 0 || err == nil {
		c.Fatalf("Bad results from 'docker rm NoSuchContainer'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	// Be really picky
	if strings.HasSuffix(stderr, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker rm'\n%s", stderr)
	}

	// docker BadCmd: stdout=empty, stderr=all, rc=0
	cmd = exec.Command(dockerBinary, "BadCmd")
	stdout, stderr, ec, err = runCommandWithStdoutStderr(cmd)
	if len(stdout) != 0 || len(stderr) == 0 || ec == 0 || err == nil {
		c.Fatalf("Bad results from 'docker BadCmd'\nec:%d\nstdout:%s\nstderr:%s\nerr:%q", ec, stdout, stderr, err)
	}
	if stderr != "docker: 'BadCmd' is not a docker command.\nSee 'docker --help'.\n" {
		c.Fatalf("Unexcepted output for 'docker badCmd'\nstderr:%s", stderr)
	}
	// Be really picky
	if strings.HasSuffix(stderr, "\n\n") {
		c.Fatalf("Should not have a blank line at the end of 'docker rm'\n%s", stderr)
	}

}
