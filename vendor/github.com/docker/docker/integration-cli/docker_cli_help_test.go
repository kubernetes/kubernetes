package main

import (
	"fmt"
	"runtime"
	"strings"
	"unicode"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/pkg/homedir"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestHelpTextVerify(c *check.C) {
	// FIXME(vdemeester) should be a unit test, probably using golden files ?
	testRequires(c, DaemonIsLinux)

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
	baseEnvs := appendBaseEnv(true)

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
		result := icmd.RunCmd(icmd.Cmd{
			Command: []string{dockerBinary, "help"},
			Env:     newEnvs,
		})
		result.Assert(c, icmd.Success)
		lines := strings.Split(result.Combined(), "\n")
		for _, line := range lines {
			// All lines should not end with a space
			c.Assert(line, checker.Not(checker.HasSuffix), " ", check.Commentf("Line should not end with a space"))

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

		// Make sure each cmd's help text fits within 90 chars and that
		// on non-windows system we use ~ when possible (to shorten things).
		// Pull the list of commands from the "Commands:" section of docker help
		// FIXME(vdemeester) Why re-run help ?
		//helpCmd = exec.Command(dockerBinary, "help")
		//helpCmd.Env = newEnvs
		//out, _, err = runCommandWithOutput(helpCmd)
		//c.Assert(err, checker.IsNil, check.Commentf(out))
		i := strings.Index(result.Combined(), "Commands:")
		c.Assert(i, checker.GreaterOrEqualThan, 0, check.Commentf("Missing 'Commands:' in:\n%s", result.Combined()))

		cmds := []string{}
		// Grab all chars starting at "Commands:"
		helpOut := strings.Split(result.Combined()[i:], "\n")
		// Skip first line, it is just "Commands:"
		helpOut = helpOut[1:]

		// Create the list of commands we want to test
		cmdsToTest := []string{}
		for _, cmd := range helpOut {
			// Stop on blank line or non-indented line
			if cmd == "" || !unicode.IsSpace(rune(cmd[0])) {
				break
			}

			// Grab just the first word of each line
			cmd = strings.Split(strings.TrimSpace(cmd), " ")[0]
			cmds = append(cmds, cmd) // Saving count for later

			cmdsToTest = append(cmdsToTest, cmd)
		}

		// Add some 'two word' commands - would be nice to automatically
		// calculate this list - somehow
		cmdsToTest = append(cmdsToTest, "volume create")
		cmdsToTest = append(cmdsToTest, "volume inspect")
		cmdsToTest = append(cmdsToTest, "volume ls")
		cmdsToTest = append(cmdsToTest, "volume rm")
		cmdsToTest = append(cmdsToTest, "network connect")
		cmdsToTest = append(cmdsToTest, "network create")
		cmdsToTest = append(cmdsToTest, "network disconnect")
		cmdsToTest = append(cmdsToTest, "network inspect")
		cmdsToTest = append(cmdsToTest, "network ls")
		cmdsToTest = append(cmdsToTest, "network rm")

		if testEnv.ExperimentalDaemon() {
			cmdsToTest = append(cmdsToTest, "checkpoint create")
			cmdsToTest = append(cmdsToTest, "checkpoint ls")
			cmdsToTest = append(cmdsToTest, "checkpoint rm")
		}

		// Divide the list of commands into go routines and  run the func testcommand on the commands in parallel
		// to save runtime of test

		errChan := make(chan error)

		for index := 0; index < len(cmdsToTest); index++ {
			go func(index int) {
				errChan <- testCommand(cmdsToTest[index], newEnvs, scanForHome, home)
			}(index)
		}

		for index := 0; index < len(cmdsToTest); index++ {
			err := <-errChan
			if err != nil {
				c.Fatal(err)
			}
		}
	}
}

func (s *DockerSuite) TestHelpExitCodesHelpOutput(c *check.C) {
	// Test to make sure the exit code and output (stdout vs stderr) of
	// various good and bad cases are what we expect

	// docker : stdout=all, stderr=empty, rc=0
	out := cli.DockerCmd(c).Combined()
	// Be really pick
	c.Assert(out, checker.Not(checker.HasSuffix), "\n\n", check.Commentf("Should not have a blank line at the end of 'docker'\n"))

	// docker help: stdout=all, stderr=empty, rc=0
	out = cli.DockerCmd(c, "help").Combined()
	// Be really pick
	c.Assert(out, checker.Not(checker.HasSuffix), "\n\n", check.Commentf("Should not have a blank line at the end of 'docker help'\n"))

	// docker --help: stdout=all, stderr=empty, rc=0
	out = cli.DockerCmd(c, "--help").Combined()
	// Be really pick
	c.Assert(out, checker.Not(checker.HasSuffix), "\n\n", check.Commentf("Should not have a blank line at the end of 'docker --help'\n"))

	// docker inspect busybox: stdout=all, stderr=empty, rc=0
	// Just making sure stderr is empty on valid cmd
	out = cli.DockerCmd(c, "inspect", "busybox").Combined()
	// Be really pick
	c.Assert(out, checker.Not(checker.HasSuffix), "\n\n", check.Commentf("Should not have a blank line at the end of 'docker inspect busyBox'\n"))

	// docker rm: stdout=empty, stderr=all, rc!=0
	// testing the min arg error msg
	cli.Docker(cli.Args("rm")).Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Out:      "",
		// Should not contain full help text but should contain info about
		// # of args and Usage line
		Err: "requires at least 1 argument",
	})

	// docker rm NoSuchContainer: stdout=empty, stderr=all, rc=0
	// testing to make sure no blank line on error
	result := cli.Docker(cli.Args("rm", "NoSuchContainer")).Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Out:      "",
	})
	// Be really picky
	c.Assert(len(result.Stderr()), checker.Not(checker.Equals), 0)
	c.Assert(result.Stderr(), checker.Not(checker.HasSuffix), "\n\n", check.Commentf("Should not have a blank line at the end of 'docker rm'\n"))

	// docker BadCmd: stdout=empty, stderr=all, rc=0
	cli.Docker(cli.Args("BadCmd")).Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Out:      "",
		Err:      "docker: 'BadCmd' is not a docker command.\nSee 'docker --help'\n",
	})
}

func testCommand(cmd string, newEnvs []string, scanForHome bool, home string) error {

	args := strings.Split(cmd+" --help", " ")

	// Check the full usage text
	result := icmd.RunCmd(icmd.Cmd{
		Command: append([]string{dockerBinary}, args...),
		Env:     newEnvs,
	})
	err := result.Error
	out := result.Stdout()
	stderr := result.Stderr()
	if len(stderr) != 0 {
		return fmt.Errorf("Error on %q help. non-empty stderr:%q\n", cmd, stderr)
	}
	if strings.HasSuffix(out, "\n\n") {
		return fmt.Errorf("Should not have blank line on %q\n", cmd)
	}
	if !strings.Contains(out, "--help") {
		return fmt.Errorf("All commands should mention '--help'. Command '%v' did not.\n", cmd)
	}

	if err != nil {
		return fmt.Errorf(out)
	}

	// Check each line for lots of stuff
	lines := strings.Split(out, "\n")
	for _, line := range lines {
		i := strings.Index(line, "~")
		if i >= 0 && i != len(line)-1 && line[i+1] != '/' {
			return fmt.Errorf("Help for %q should not have used ~:\n%s", cmd, line)
		}

		// Options should NOT end with a period
		if strings.HasPrefix(line, "  -") && strings.HasSuffix(line, ".") {
			return fmt.Errorf("Help for %q should not end with a period: %s", cmd, line)
		}

		// Options should NOT end with a space
		if strings.HasSuffix(line, " ") {
			return fmt.Errorf("Help for %q should not end with a space: %s", cmd, line)
		}

	}

	// For each command make sure we generate an error
	// if we give a bad arg
	args = strings.Split(cmd+" --badArg", " ")

	out, _, err = dockerCmdWithError(args...)
	if err == nil {
		return fmt.Errorf(out)
	}

	// Be really picky
	if strings.HasSuffix(stderr, "\n\n") {
		return fmt.Errorf("Should not have a blank line at the end of 'docker rm'\n")
	}

	// Now make sure that each command will print a short-usage
	// (not a full usage - meaning no opts section) if we
	// are missing a required arg or pass in a bad arg

	// These commands will never print a short-usage so don't test
	noShortUsage := map[string]string{
		"images":        "",
		"login":         "",
		"logout":        "",
		"network":       "",
		"stats":         "",
		"volume create": "",
	}

	if _, ok := noShortUsage[cmd]; !ok {
		// skipNoArgs are ones that we don't want to try w/o
		// any args. Either because it'll hang the test or
		// lead to incorrect test result (like false negative).
		// Whatever the reason, skip trying to run w/o args and
		// jump to trying with a bogus arg.
		skipNoArgs := map[string]struct{}{
			"daemon": {},
			"events": {},
			"load":   {},
		}

		var result *icmd.Result
		if _, ok := skipNoArgs[cmd]; !ok {
			result = dockerCmdWithResult(strings.Split(cmd, " ")...)
		}

		// If its ok w/o any args then try again with an arg
		if result == nil || result.ExitCode == 0 {
			result = dockerCmdWithResult(strings.Split(cmd+" badArg", " ")...)
		}

		if err := result.Compare(icmd.Expected{
			Out:      icmd.None,
			Err:      "\nUsage:",
			ExitCode: 1,
		}); err != nil {
			return err
		}

		stderr := result.Stderr()
		// Shouldn't have full usage
		if strings.Contains(stderr, "--help=false") {
			return fmt.Errorf("Should not have full usage on %q:%v", result.Cmd.Args, stderr)
		}
		if strings.HasSuffix(stderr, "\n\n") {
			return fmt.Errorf("Should not have a blank line on %q\n%v", result.Cmd.Args, stderr)
		}
	}

	return nil
}
