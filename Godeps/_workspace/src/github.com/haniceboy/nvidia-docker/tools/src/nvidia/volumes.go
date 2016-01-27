// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package nvidia

import (
	"bufio"
	"bytes"
	"debug/elf"
	"encoding/binary"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/haniceboy/nvidia-docker/tools/src/ldcache"
)

const (
	binDir   = "bin"
	lib32Dir = "lib"
	lib64Dir = "lib64"
)

type components map[string][]string

type volumeDir struct {
	name  string
	files []string
}

type VolumeInfo struct {
	Name       string
	Mountpoint string
	Components components
}

type Volume struct {
	*VolumeInfo

	Path    string
	Version string
	dirs    []volumeDir
}

type VolumeMap map[string]*Volume

var Volumes = []VolumeInfo{
	{
		"nvidia_driver",
		"/usr/local/nvidia",
		components{
			"binaries": {
				//"nvidia-modprobe",       // Kernel module loader
				//"nvidia-settings",       // X server settings
				//"nvidia-xconfig",        // X xorg.conf editor
				"nvidia-cuda-mps-control", // Multi process service CLI
				"nvidia-cuda-mps-server",  // Multi process service server
				"nvidia-debugdump",        // GPU coredump utility
				"nvidia-persistenced",     // Persistence mode utility
				"nvidia-smi",              // System management interface
			},
			"libraries": {
				//"libnvidia-cfg.so",  // GPU configuration (used by nvidia-xconfig)
				//"libnvidia-gtk2.so", // GTK2 (used by nvidia-settings)
				//"libnvidia-gtk3.so", // GTK3 (used by nvidia-settings)
				//"libnvidia-wfb.so",  // Wrapped software rendering

				"libnvidia-ml.so", // Management library
				"libcuda.so",      // CUDA runtime

				//"libvdpau.so",       // VDPAU ICD loader
				//"libvdpau_trace.so", // VDPAU debug trace
				"libvdpau_nvidia.so",  // NVIDIA VDPAU ICD
				"libnvidia-encode.so", // Video encoder
				"libnvcuvid.so",       // Video decoder

				"libnvidia-fbc.so", // Framebuffer capture
				"libnvidia-ifr.so", // OpenGL framebuffer capture

				//"libOpenCL.so",        // OpenCL ICD loader
				"libnvidia-opencl.so",   // NVIDIA OpenCL ICD
				"libnvidia-compiler.so", // PTX compiler (used by libnvidia-opencl)

				//"libglx.so",         // GLX extension module for X server
				"libGL.so",            // OpenGL / GLX
				"libnvidia-glcore.so", // OpenGL core (used by libGL and libglx)
				"libnvidia-tls.so",    // Thread local storage (used by libGL and libglx)

				//"libOpenGL.so",       // OpenGL ICD loader
				//"libEGL.so",          // EGL ICD loader
				//"libGLdispatch.so",   // OpenGL dispatch (used by libOpenGL and libEGL)
				"libEGL_nvidia.so",     // NVIDIA EGL ICD
				"libGLESv1_CM.so",      // OpenGL ES v1 common profile
				"libGLESv2.so",         // OpenGL ES v2
				"libnvidia-eglcore.so", // EGL core (used by libGLES and libEGL_nvidia)
				"libnvidia-glsi.so",    // OpenGL system interaction (used by libEGL_nvidia)
			},
		},
	},
}

func blacklisted(file string, obj *elf.File) (bool, error) {
	lib := regexp.MustCompile("^.*/lib([\\w-]+)\\.so[\\d.]*$")
	glcore := regexp.MustCompile("libnvidia-e?glcore.so")

	if m := lib.FindStringSubmatch(file); m != nil {
		switch m[1] {

		// Blacklist EGL/OpenGL libraries issued by other vendors
		case "GLESv1_CM":
			fallthrough
		case "GLESv2":
			fallthrough
		case "GL":
			deps, err := obj.DynString(elf.DT_NEEDED)
			if err != nil {
				return false, err
			}
			for _, d := range deps {
				if glcore.MatchString(d) {
					return false, nil
				}
			}
			return true, nil

		// Blacklist TLS libraries using the old ABI (!= 2.3.99)
		case "nvidia-tls":
			const abi = 0x6300000003
			s, err := obj.Section(".note.ABI-tag").Data()
			if err != nil {
				return false, err
			}
			return binary.LittleEndian.Uint64(s[24:]) != abi, nil
		}
	}
	return false, nil
}

func (v *Volume) CreateAt(path string) error {
	v.Path = path
	v.Version = ""
	return v.Create()
}

func (v *Volume) Create() (err error) {
	if err = os.MkdirAll(v.Path, 0755); err != nil {
		return
	}
	defer func() {
		if err != nil {
			v.Remove()
		}
	}()

	for _, d := range v.dirs {
		dir := path.Join(v.Path, v.Version, d.name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
		for _, f := range d.files {
			obj, err := elf.Open(f)
			if err != nil {
				return err
			}
			defer obj.Close()

			ok, err := blacklisted(f, obj)
			if err != nil {
				return err
			}
			if ok {
				continue
			}

			l := path.Join(dir, path.Base(f))
			if err := os.Link(f, l); err != nil {
				return err
			}
			soname, err := obj.DynString(elf.DT_SONAME)
			if err != nil {
				return err
			}
			if len(soname) > 0 {
				f = path.Join(v.Mountpoint, d.name, path.Base(f))
				l = path.Join(dir, soname[0])
				if err := os.Symlink(f, l); err != nil &&
					!os.IsExist(err.(*os.LinkError).Err) {
					return err
				}
			}
		}
	}
	return nil
}

func (v *Volume) Remove(version ...string) error {
	vv := v.Version
	if len(version) == 1 {
		vv = version[0]
	}
	return os.RemoveAll(path.Join(v.Path, vv))
}

func (v *Volume) Exists() (bool, error) {
	p := path.Join(v.Path, v.Version)
	_, err := os.Stat(p)
	if os.IsNotExist(err) {
		return false, nil
	}
	return true, err
}

func which(bins ...string) ([]string, error) {
	paths := make([]string, 0, len(bins))

	out, _ := exec.Command("which", bins...).Output()
	r := bufio.NewReader(bytes.NewBuffer(out))
	for {
		p, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if p = strings.TrimSpace(p); !path.IsAbs(p) {
			continue
		}
		path, err := filepath.EvalSymlinks(p)
		if err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}

func LookupVolumes(prefix string) (vols VolumeMap, err error) {
	drv, err := GetDriverVersion()
	if err != nil {
		return nil, err
	}
	cache, err := ldcache.Open()
	if err != nil {
		return nil, err
	}
	defer func() {
		if e := cache.Close(); err == nil {
			err = e
		}
	}()

	vols = make(VolumeMap, len(Volumes))

	for i := range Volumes {
		vol := &Volume{
			VolumeInfo: &Volumes[i],
			Path:       path.Join(prefix, Volumes[i].Name),
			Version:    drv,
		}

		for t, c := range vol.Components {
			switch t {
			case "binaries":
				bins, err := which(c...)
				if err != nil {
					return nil, err
				}
				vol.dirs = append(vol.dirs, volumeDir{binDir, bins})
			case "libraries":
				libs32, libs64 := cache.Lookup(c...)
				vol.dirs = append(vol.dirs,
					volumeDir{lib32Dir, libs32},
					volumeDir{lib64Dir, libs64},
				)
			}
		}
		vols[vol.Name] = vol
	}
	return
}
