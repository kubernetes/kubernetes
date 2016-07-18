package client

import (
	"errors"
	"fmt"
)

// ErrConnectionFailed is an error raised when the connection between the client and the server failed.
var ErrConnectionFailed = errors.New("Cannot connect to the Docker daemon. Is the docker daemon running on this host?")

type notFound interface {
	error
	NotFound() bool // Is the error a NotFound error
}

// IsErrNotFound returns true if the error is caused with an
// object (image, container, network, volume, …) is not found in the docker host.
func IsErrNotFound(err error) bool {
	te, ok := err.(notFound)
	return ok && te.NotFound()
}

// imageNotFoundError implements an error returned when an image is not in the docker host.
type imageNotFoundError struct {
	imageID string
}

// NoFound indicates that this error type is of NotFound
func (e imageNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of an imageNotFoundError
func (e imageNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such image: %s", e.imageID)
}

// IsErrImageNotFound returns true if the error is caused
// when an image is not found in the docker host.
func IsErrImageNotFound(err error) bool {
	return IsErrNotFound(err)
}

// containerNotFoundError implements an error returned when a container is not in the docker host.
type containerNotFoundError struct {
	containerID string
}

// NoFound indicates that this error type is of NotFound
func (e containerNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of a containerNotFoundError
func (e containerNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such container: %s", e.containerID)
}

// IsErrContainerNotFound returns true if the error is caused
// when a container is not found in the docker host.
func IsErrContainerNotFound(err error) bool {
	return IsErrNotFound(err)
}

// networkNotFoundError implements an error returned when a network is not in the docker host.
type networkNotFoundError struct {
	networkID string
}

// NoFound indicates that this error type is of NotFound
func (e networkNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of a networkNotFoundError
func (e networkNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such network: %s", e.networkID)
}

// IsErrNetworkNotFound returns true if the error is caused
// when a network is not found in the docker host.
func IsErrNetworkNotFound(err error) bool {
	return IsErrNotFound(err)
}

// volumeNotFoundError implements an error returned when a volume is not in the docker host.
type volumeNotFoundError struct {
	volumeID string
}

// NoFound indicates that this error type is of NotFound
func (e volumeNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of a networkNotFoundError
func (e volumeNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such volume: %s", e.volumeID)
}

// IsErrVolumeNotFound returns true if the error is caused
// when a volume is not found in the docker host.
func IsErrVolumeNotFound(err error) bool {
	return IsErrNotFound(err)
}

// unauthorizedError represents an authorization error in a remote registry.
type unauthorizedError struct {
	cause error
}

// Error returns a string representation of an unauthorizedError
func (u unauthorizedError) Error() string {
	return u.cause.Error()
}

// IsErrUnauthorized returns true if the error is caused
// when a remote registry authentication fails
func IsErrUnauthorized(err error) bool {
	_, ok := err.(unauthorizedError)
	return ok
}

// nodeNotFoundError implements an error returned when a node is not found.
type nodeNotFoundError struct {
	nodeID string
}

// Error returns a string representation of a nodeNotFoundError
func (e nodeNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such node: %s", e.nodeID)
}

// NoFound indicates that this error type is of NotFound
func (e nodeNotFoundError) NotFound() bool {
	return true
}

// IsErrNodeNotFound returns true if the error is caused
// when a node is not found.
func IsErrNodeNotFound(err error) bool {
	_, ok := err.(nodeNotFoundError)
	return ok
}

// serviceNotFoundError implements an error returned when a service is not found.
type serviceNotFoundError struct {
	serviceID string
}

// Error returns a string representation of a serviceNotFoundError
func (e serviceNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such service: %s", e.serviceID)
}

// NoFound indicates that this error type is of NotFound
func (e serviceNotFoundError) NotFound() bool {
	return true
}

// IsErrServiceNotFound returns true if the error is caused
// when a service is not found.
func IsErrServiceNotFound(err error) bool {
	_, ok := err.(serviceNotFoundError)
	return ok
}

// taskNotFoundError implements an error returned when a task is not found.
type taskNotFoundError struct {
	taskID string
}

// Error returns a string representation of a taskNotFoundError
func (e taskNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such task: %s", e.taskID)
}

// NoFound indicates that this error type is of NotFound
func (e taskNotFoundError) NotFound() bool {
	return true
}

// IsErrTaskNotFound returns true if the error is caused
// when a task is not found.
func IsErrTaskNotFound(err error) bool {
	_, ok := err.(taskNotFoundError)
	return ok
}

type pluginPermissionDenied struct {
	name string
}

func (e pluginPermissionDenied) Error() string {
	return "Permission denied while installing plugin " + e.name
}

// IsErrPluginPermissionDenied returns true if the error is caused
// when a user denies a plugin's permissions
func IsErrPluginPermissionDenied(err error) bool {
	_, ok := err.(pluginPermissionDenied)
	return ok
}
