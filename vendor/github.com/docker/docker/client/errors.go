package client

import (
	"fmt"

	"github.com/docker/docker/api/types/versions"
	"github.com/pkg/errors"
)

// errConnectionFailed implements an error returned when connection failed.
type errConnectionFailed struct {
	host string
}

// Error returns a string representation of an errConnectionFailed
func (err errConnectionFailed) Error() string {
	if err.host == "" {
		return "Cannot connect to the Docker daemon. Is the docker daemon running on this host?"
	}
	return fmt.Sprintf("Cannot connect to the Docker daemon at %s. Is the docker daemon running?", err.host)
}

// IsErrConnectionFailed returns true if the error is caused by connection failed.
func IsErrConnectionFailed(err error) bool {
	_, ok := errors.Cause(err).(errConnectionFailed)
	return ok
}

// ErrorConnectionFailed returns an error with host in the error message when connection to docker daemon failed.
func ErrorConnectionFailed(host string) error {
	return errConnectionFailed{host: host}
}

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

// NotFound indicates that this error type is of NotFound
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

// NotFound indicates that this error type is of NotFound
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

// NotFound indicates that this error type is of NotFound
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

// NotFound indicates that this error type is of NotFound
func (e volumeNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of a volumeNotFoundError
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

// NotFound indicates that this error type is of NotFound
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

// NotFound indicates that this error type is of NotFound
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

// NotFound indicates that this error type is of NotFound
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

// NewVersionError returns an error if the APIVersion required
// if less than the current supported version
func (cli *Client) NewVersionError(APIrequired, feature string) error {
	if cli.version != "" && versions.LessThan(cli.version, APIrequired) {
		return fmt.Errorf("%q requires API version %s, but the Docker daemon API version is %s", feature, APIrequired, cli.version)
	}
	return nil
}

// secretNotFoundError implements an error returned when a secret is not found.
type secretNotFoundError struct {
	name string
}

// Error returns a string representation of a secretNotFoundError
func (e secretNotFoundError) Error() string {
	return fmt.Sprintf("Error: no such secret: %s", e.name)
}

// NotFound indicates that this error type is of NotFound
func (e secretNotFoundError) NotFound() bool {
	return true
}

// IsErrSecretNotFound returns true if the error is caused
// when a secret is not found.
func IsErrSecretNotFound(err error) bool {
	_, ok := err.(secretNotFoundError)
	return ok
}

// configNotFoundError implements an error returned when a config is not found.
type configNotFoundError struct {
	name string
}

// Error returns a string representation of a configNotFoundError
func (e configNotFoundError) Error() string {
	return fmt.Sprintf("Error: no such config: %s", e.name)
}

// NotFound indicates that this error type is of NotFound
func (e configNotFoundError) NotFound() bool {
	return true
}

// IsErrConfigNotFound returns true if the error is caused
// when a config is not found.
func IsErrConfigNotFound(err error) bool {
	_, ok := err.(configNotFoundError)
	return ok
}

// pluginNotFoundError implements an error returned when a plugin is not in the docker host.
type pluginNotFoundError struct {
	name string
}

// NotFound indicates that this error type is of NotFound
func (e pluginNotFoundError) NotFound() bool {
	return true
}

// Error returns a string representation of a pluginNotFoundError
func (e pluginNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such plugin: %s", e.name)
}

// IsErrPluginNotFound returns true if the error is caused
// when a plugin is not found in the docker host.
func IsErrPluginNotFound(err error) bool {
	return IsErrNotFound(err)
}
