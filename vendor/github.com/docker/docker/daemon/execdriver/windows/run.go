// +build windows

package windows

// Note this is alpha code for the bring up of containers on Windows.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/microsoft/hcsshim"
	"github.com/natefinch/npipe"
)

type layer struct {
	Id   string
	Path string
}

type defConfig struct {
	DefFile string
}

type networkConnection struct {
	NetworkName string
	EnableNat   bool
}
type networkSettings struct {
	MacAddress string
}

type device struct {
	DeviceType string
	Connection interface{}
	Settings   interface{}
}

type containerInit struct {
	SystemType              string
	Name                    string
	IsDummy                 bool
	VolumePath              string
	Devices                 []device
	IgnoreFlushesDuringBoot bool
	LayerFolderPath         string
	Layers                  []layer
}

func (d *driver) Run(c *execdriver.Command, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (execdriver.ExitStatus, error) {

	var (
		term                           execdriver.Terminal
		err                            error
		inListen, outListen, errListen *npipe.PipeListener
	)

	// Make sure the client isn't asking for options which aren't supported
	err = checkSupportedOptions(c)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	cu := &containerInit{
		SystemType:              "Container",
		Name:                    c.ID,
		IsDummy:                 dummyMode,
		VolumePath:              c.Rootfs,
		IgnoreFlushesDuringBoot: c.FirstStart,
		LayerFolderPath:         c.LayerFolder,
	}

	for i := 0; i < len(c.LayerPaths); i++ {
		cu.Layers = append(cu.Layers, layer{
			Id:   hcsshim.NewGUID(c.LayerPaths[i]).ToString(),
			Path: c.LayerPaths[i],
		})
	}

	if c.Network.Interface != nil {
		dev := device{
			DeviceType: "Network",
			Connection: &networkConnection{
				NetworkName: c.Network.Interface.Bridge,
				EnableNat:   false,
			},
		}

		if c.Network.Interface.MacAddress != "" {
			windowsStyleMAC := strings.Replace(
				c.Network.Interface.MacAddress, ":", "-", -1)
			dev.Settings = networkSettings{
				MacAddress: windowsStyleMAC,
			}
		}

		logrus.Debugf("Virtual switch '%s', mac='%s'", c.Network.Interface.Bridge, c.Network.Interface.MacAddress)

		cu.Devices = append(cu.Devices, dev)
	} else {
		logrus.Debugln("No network interface")
	}

	configurationb, err := json.Marshal(cu)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	configuration := string(configurationb)

	err = hcsshim.CreateComputeSystem(c.ID, configuration)
	if err != nil {
		logrus.Debugln("Failed to create temporary container ", err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	// Start the container
	logrus.Debugln("Starting container ", c.ID)
	err = hcsshim.StartComputeSystem(c.ID)
	if err != nil {
		logrus.Errorf("Failed to start compute system: %s", err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	defer func() {
		// Stop the container

		if terminateMode {
			logrus.Debugf("Terminating container %s", c.ID)
			if err := hcsshim.TerminateComputeSystem(c.ID); err != nil {
				// IMPORTANT: Don't fail if fails to change state. It could already
				// have been stopped through kill().
				// Otherwise, the docker daemon will hang in job wait()
				logrus.Warnf("Ignoring error from TerminateComputeSystem %s", err)
			}
		} else {
			logrus.Debugf("Shutting down container %s", c.ID)
			if err := hcsshim.ShutdownComputeSystem(c.ID); err != nil {
				// IMPORTANT: Don't fail if fails to change state. It could already
				// have been stopped through kill().
				// Otherwise, the docker daemon will hang in job wait()
				logrus.Warnf("Ignoring error from ShutdownComputeSystem %s", err)
			}
		}
	}()

	// We use a different pipe name between real and dummy mode in the HCS
	var serverPipeFormat, clientPipeFormat string
	if dummyMode {
		clientPipeFormat = `\\.\pipe\docker-run-%[1]s-%[2]s`
		serverPipeFormat = clientPipeFormat
	} else {
		clientPipeFormat = `\\.\pipe\docker-run-%[2]s`
		serverPipeFormat = `\\.\Containers\%[1]s\Device\NamedPipe\docker-run-%[2]s`
	}

	createProcessParms := hcsshim.CreateProcessParams{
		EmulateConsole:   c.ProcessConfig.Tty,
		WorkingDirectory: c.WorkingDir,
		ConsoleSize:      c.ProcessConfig.ConsoleSize,
	}

	// Configure the environment for the process
	createProcessParms.Environment = setupEnvironmentVariables(c.ProcessConfig.Env)

	// Connect stdin
	if pipes.Stdin != nil {
		stdInPipe := fmt.Sprintf(serverPipeFormat, c.ID, "stdin")
		createProcessParms.StdInPipe = fmt.Sprintf(clientPipeFormat, c.ID, "stdin")

		// Listen on the named pipe
		inListen, err = npipe.Listen(stdInPipe)
		if err != nil {
			logrus.Errorf("stdin failed to listen on %s err=%s", stdInPipe, err)
			return execdriver.ExitStatus{ExitCode: -1}, err
		}
		defer inListen.Close()

		// Launch a goroutine to do the accept. We do this so that we can
		// cause an otherwise blocking goroutine to gracefully close when
		// the caller (us) closes the listener
		go stdinAccept(inListen, stdInPipe, pipes.Stdin)
	}

	// Connect stdout
	stdOutPipe := fmt.Sprintf(serverPipeFormat, c.ID, "stdout")
	createProcessParms.StdOutPipe = fmt.Sprintf(clientPipeFormat, c.ID, "stdout")

	outListen, err = npipe.Listen(stdOutPipe)
	if err != nil {
		logrus.Errorf("stdout failed to listen on %s err=%s", stdOutPipe, err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	defer outListen.Close()
	go stdouterrAccept(outListen, stdOutPipe, pipes.Stdout)

	// No stderr on TTY.
	if !c.ProcessConfig.Tty {
		// Connect stderr
		stdErrPipe := fmt.Sprintf(serverPipeFormat, c.ID, "stderr")
		createProcessParms.StdErrPipe = fmt.Sprintf(clientPipeFormat, c.ID, "stderr")
		errListen, err = npipe.Listen(stdErrPipe)
		if err != nil {
			logrus.Errorf("stderr failed to listen on %s err=%s", stdErrPipe, err)
			return execdriver.ExitStatus{ExitCode: -1}, err
		}
		defer errListen.Close()
		go stdouterrAccept(errListen, stdErrPipe, pipes.Stderr)
	}

	// This should get caught earlier, but just in case - validate that we
	// have something to run
	if c.ProcessConfig.Entrypoint == "" {
		err = errors.New("No entrypoint specified")
		logrus.Error(err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	// Build the command line of the process
	createProcessParms.CommandLine = c.ProcessConfig.Entrypoint
	for _, arg := range c.ProcessConfig.Arguments {
		logrus.Debugln("appending ", arg)
		createProcessParms.CommandLine += " " + arg
	}
	logrus.Debugf("CommandLine: %s", createProcessParms.CommandLine)

	// Start the command running in the container.
	var pid uint32
	pid, err = hcsshim.CreateProcessInComputeSystem(c.ID, createProcessParms)

	if err != nil {
		logrus.Errorf("CreateProcessInComputeSystem() failed %s", err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	//Save the PID as we'll need this in Kill()
	logrus.Debugf("PID %d", pid)
	c.ContainerPid = int(pid)

	if c.ProcessConfig.Tty {
		term = NewTtyConsole(c.ID, pid)
	} else {
		term = NewStdConsole()
	}
	c.ProcessConfig.Terminal = term

	// Maintain our list of active containers. We'll need this later for exec
	// and other commands.
	d.Lock()
	d.activeContainers[c.ID] = &activeContainer{
		command: c,
	}
	d.Unlock()

	// Invoke the start callback
	if startCallback != nil {
		startCallback(&c.ProcessConfig, int(pid))
	}

	var exitCode int32
	exitCode, err = hcsshim.WaitForProcessInComputeSystem(c.ID, pid)
	if err != nil {
		logrus.Errorf("Failed to WaitForProcessInComputeSystem %s", err)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	logrus.Debugf("Exiting Run() exitCode %d id=%s", exitCode, c.ID)
	return execdriver.ExitStatus{ExitCode: int(exitCode)}, nil
}
