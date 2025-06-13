package utils

import "path/filepath"

const (
	sysfsPath = "/sys"
)

// Sysfs provides methods to access various sysfs paths.
// It abstracts the sysfs directory structure, allowing for easy retrieval of paths
// ref: https://linux-kernel-labs.github.io/refs/heads/master/labs/device_model.html#sysfs
type Sysfs interface {
	// Devices returns the full path to the devices directory with the specified path appended.
	// The path is relative to /sys/devices.
	Devices(path string) string

	// Bus returns the full path to the bus directory with the specified path appended.
	// The path is relative to /sys/bus.
	Bus(path string) string

	// Block returns the full path to the block directory with the specified path appended.
	// The path is relative to /sys/block.
	Block(path string) string

	// Class returns the full path to the class directory with the specified path appended.
	// The path is relative to /sys/class.
	Class(path string) string

	// Dev returns the full path to the dev directory with the specified path appended.
	// The path is relative to /sys/dev.
	Dev(path string) string

	// Firmware returns the full path to the firmware directory with the specified path appended.
	// The path is relative to /sys/firmware.
	Firmware(path string) string

	// Kernel returns the full path to the kernel directory with the specified path appended.
	// The path is relative to /sys/kernel.
	Kernel(path string) string

	// Module returns the full path to the module directory with the specified path appended.
	// The path is relative to /sys/module.
	Module(path string) string
}

// sysfs provides methods to access sysfs paths
// root is the root path of the sysfs filesystem, typically "/sys".
// Setting non-"/sys" root is only for testing purposes.
type sysfs struct {
	root string
}

// NewSysfs creates a new Sysfs instance(/sys)
func NewSysfs() Sysfs {
	return &sysfs{root: sysfsPath}
}

// NewSysfsWithRoot creates a new Sysfs instance with the specified root path.
// If the root path is empty, it defaults to "/sys".
// This method is only for testing purposes.
func NewSysfsWithRoot(root string) Sysfs {
	if root == "" {
		root = sysfsPath
	}
	return &sysfs{root: root}
}

func (s *sysfs) Devices(path string) string {
	return filepath.Join(s.root, "devices", path)
}

func (s *sysfs) Bus(path string) string {
	return filepath.Join(s.root, "bus", path)
}

func (s *sysfs) Block(path string) string {
	return filepath.Join(s.root, "block", path)
}

func (s *sysfs) Class(path string) string {
	return filepath.Join(s.root, "class", path)
}

func (s *sysfs) Dev(path string) string {
	return filepath.Join(s.root, "dev", path)
}

func (s *sysfs) Firmware(path string) string {
	return filepath.Join(s.root, "firmware", path)
}
func (s *sysfs) Kernel(path string) string {
	return filepath.Join(s.root, "kernel", path)
}

func (s *sysfs) Module(path string) string {
	return filepath.Join(s.root, "module", path)
}
