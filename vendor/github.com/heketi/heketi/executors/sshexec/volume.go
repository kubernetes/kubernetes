//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package sshexec

import (
	"encoding/xml"
	"fmt"

	"github.com/heketi/heketi/executors"
	"github.com/lpabon/godbc"
)

func (s *SshExecutor) VolumeCreate(host string,
	volume *executors.VolumeRequest) (*executors.Volume, error) {

	godbc.Require(volume != nil)
	godbc.Require(host != "")
	godbc.Require(len(volume.Bricks) > 0)
	godbc.Require(volume.Name != "")

	cmd := fmt.Sprintf("gluster --mode=script volume create %v ", volume.Name)

	var (
		inSet     int
		maxPerSet int
	)
	switch volume.Type {
	case executors.DurabilityNone:
		logger.Info("Creating volume %v with no durability", volume.Name)
		inSet = 1
		maxPerSet = 15
	case executors.DurabilityReplica:
		logger.Info("Creating volume %v replica %v", volume.Name, volume.Replica)
		cmd += fmt.Sprintf("replica %v ", volume.Replica)
		inSet = volume.Replica
		maxPerSet = 5
	case executors.DurabilityDispersion:
		logger.Info("Creating volume %v dispersion %v+%v",
			volume.Name, volume.Data, volume.Redundancy)
		cmd += fmt.Sprintf("disperse-data %v redundancy %v ", volume.Data, volume.Redundancy)
		inSet = volume.Data + volume.Redundancy
		maxPerSet = 1
	}

	// There could many, many bricks, which could render a single command
	// line that creates the volume with all the bricks too long.
	// Therefore, we initially create the volume with the first brick set
	// only, and then add each brick set in one subsequent command.

	for _, brick := range volume.Bricks[:inSet] {
		cmd += fmt.Sprintf("%v:%v ", brick.Host, brick.Path)
	}

	commands := []string{cmd}

	commands = append(commands, s.createAddBrickCommands(volume, inSet, inSet, maxPerSet)...)

	commands = append(commands, s.createVolumeOptionsCommand(volume)...)

	commands = append(commands, fmt.Sprintf("gluster --mode=script volume start %v", volume.Name))

	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		s.VolumeDestroy(host, volume.Name)
		return nil, err
	}

	return &executors.Volume{}, nil
}

func (s *SshExecutor) VolumeExpand(host string,
	volume *executors.VolumeRequest) (*executors.Volume, error) {

	godbc.Require(volume != nil)
	godbc.Require(host != "")
	godbc.Require(len(volume.Bricks) > 0)
	godbc.Require(volume.Name != "")

	var (
		inSet     int
		maxPerSet int
	)
	switch volume.Type {
	case executors.DurabilityNone:
		inSet = 1
		maxPerSet = 15
	case executors.DurabilityReplica:
		inSet = volume.Replica
		maxPerSet = 5
	case executors.DurabilityDispersion:
		inSet = volume.Data + volume.Redundancy
		maxPerSet = 1
	}

	commands := s.createAddBrickCommands(volume,
		0, // start at the beginning of the brick list
		inSet,
		maxPerSet)

	if s.RemoteExecutor.RebalanceOnExpansion() {
		commands = append(commands,
			fmt.Sprintf("gluster --mode=script volume rebalance %v start", volume.Name))
	}

	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		return nil, err
	}

	return &executors.Volume{}, nil
}

func (s *SshExecutor) VolumeDestroy(host string, volume string) error {
	godbc.Require(host != "")
	godbc.Require(volume != "")

	// First stop the volume, then delete it

	commands := []string{
		fmt.Sprintf("gluster --mode=script volume stop %v force", volume),
	}

	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		logger.LogError("Unable to stop volume %v: %v", volume, err)
	}

	commands = []string{
		fmt.Sprintf("gluster --mode=script volume delete %v", volume),
	}

	_, err = s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		return logger.Err(fmt.Errorf("Unable to delete volume %v: %v", volume, err))
	}

	return nil
}

func (s *SshExecutor) VolumeDestroyCheck(host, volume string) error {
	godbc.Require(host != "")
	godbc.Require(volume != "")

	// Determine if the volume is able to be deleted
	err := s.checkForSnapshots(host, volume)
	if err != nil {
		return err
	}

	return nil
}

func (s *SshExecutor) createVolumeOptionsCommand(volume *executors.VolumeRequest) []string {
	godbc.Require(len(volume.GlusterVolumeOptions) > 0)

	commands := []string{}
	var cmd string

	// Go through all the Options and create volume set command
	for _, volOption := range volume.GlusterVolumeOptions {
		if volOption != "" {
			cmd = fmt.Sprintf("gluster --mode=script volume set %v %v", volume.Name, volOption)
			commands = append(commands, cmd)
		}

	}
	return commands
}

func (s *SshExecutor) createAddBrickCommands(volume *executors.VolumeRequest,
	start, inSet, maxPerSet int) []string {

	commands := []string{}
	var cmd string

	// Go through all the bricks and create add-brick commands
	for index, brick := range volume.Bricks[start:] {
		if index%(inSet*maxPerSet) == 0 {
			if cmd != "" {
				// Add add-brick command to the command list
				commands = append(commands, cmd)
			}

			// Create a new add-brick command
			cmd = fmt.Sprintf("gluster --mode=script volume add-brick %v ", volume.Name)
		}

		// Add this brick to the add-brick command
		cmd += fmt.Sprintf("%v:%v ", brick.Host, brick.Path)
	}

	// Add the last add-brick command to the command list
	commands = append(commands, cmd)

	return commands
}

func (s *SshExecutor) checkForSnapshots(host, volume string) error {

	// Structure used to unmarshal XML from snapshot gluster cli
	type CliOutput struct {
		SnapList struct {
			Count int `xml:"count"`
		} `xml:"snapList"`
	}

	commands := []string{
		fmt.Sprintf("gluster --mode=script snapshot list %v --xml", volume),
	}

	output, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		return fmt.Errorf("Unable to get snapshot information from volume %v: %v", volume, err)
	}

	var snapInfo CliOutput
	err = xml.Unmarshal([]byte(output[0]), &snapInfo)
	if err != nil {
		return fmt.Errorf("Unable to determine snapshot information from volume %v: %v", volume, err)
	}

	if snapInfo.SnapList.Count > 0 {
		return fmt.Errorf("Unable to delete volume %v because it contains %v snapshots",
			volume, snapInfo.SnapList.Count)
	}

	return nil
}

func (s *SshExecutor) VolumeInfo(host string, volume string) (*executors.Volume, error) {

	godbc.Require(volume != "")
	godbc.Require(host != "")

	type CliOutput struct {
		OpRet    int               `xml:"opRet"`
		OpErrno  int               `xml:"opErrno"`
		OpErrStr string            `xml:"opErrstr"`
		VolInfo  executors.VolInfo `xml:"volInfo"`
	}

	command := []string{
		fmt.Sprintf("gluster --mode=script volume info %v --xml", volume),
	}

	//Get the xml output of volume info
	output, err := s.RemoteExecutor.RemoteCommandExecute(host, command, 10)
	if err != nil {
		return nil, fmt.Errorf("Unable to get volume info of volume name: %v", volume)
	}
	var volumeInfo CliOutput
	err = xml.Unmarshal([]byte(output[0]), &volumeInfo)
	if err != nil {
		return nil, fmt.Errorf("Unable to determine volume info of volume name: %v", volume)
	}
	logger.Debug("%+v\n", volumeInfo)
	return &volumeInfo.VolInfo.Volumes.VolumeList[0], nil
}

func (s *SshExecutor) VolumeReplaceBrick(host string, volume string, oldBrick *executors.BrickInfo, newBrick *executors.BrickInfo) error {
	godbc.Require(volume != "")
	godbc.Require(host != "")
	godbc.Require(oldBrick != nil)
	godbc.Require(newBrick != nil)

	// Replace the brick
	command := []string{
		fmt.Sprintf("gluster --mode=script volume replace-brick %v %v:%v %v:%v commit force", volume, oldBrick.Host, oldBrick.Path, newBrick.Host, newBrick.Path),
	}
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, command, 10)
	if err != nil {
		return logger.Err(fmt.Errorf("Unable to replace brick %v:%v with %v:%v for volume %v", oldBrick.Host, oldBrick.Path, newBrick.Host, newBrick.Path, volume))
	}

	return nil

}
