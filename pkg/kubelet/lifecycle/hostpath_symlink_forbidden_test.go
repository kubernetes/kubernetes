package lifecycle

import (
	"fmt"
	"io/ioutil"
	"k8s.io/api/core/v1"
	"os"
	"path"
	"testing"
)

const defaultPerm = 0750

func TestHostpathSymlinkForbidden(t *testing.T) {
	sf := NewHostpathSymlinkForbidden()
	tests := []struct {
		name          string
		prepareVol    func(pod *v1.Pod, base string) error
		shouldFail    bool
		expectedAdmit bool
	}{
		{
			name: "pod without vol",
			prepareVol: func(pod *v1.Pod, base string) error {
				return nil
			},
			shouldFail:    false,
			expectedAdmit: true,
		},
		{
			name: "pod with link dir vol",
			prepareVol: func(pod *v1.Pod, base string) error {
				origin := path.Join(base, "original")
				link := path.Join(base, "link")
				if err := os.Mkdir(origin, defaultPerm); err != nil {
					return err
				}

				if err := os.Symlink(origin, link); err != nil {
					return err
				}
				linkVol := v1.Volume{
					Name: "linkVol",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: link,
						},
					},
				}
				pod.Spec.Volumes = append(pod.Spec.Volumes, []v1.Volume{linkVol}...)
				return nil
			},
			shouldFail:    false,
			expectedAdmit: false,
		},
		{
			name: "pod with link file vol",
			prepareVol: func(pod *v1.Pod, base string) error {
				origin := path.Join(base, "original.txt")
				link := path.Join(base, "link.txt")

				err := ioutil.WriteFile(origin, []byte("hello world"), 0600)
				if err != nil {
					return err
				}

				if err = os.Symlink(origin, link); err != nil {
					return err
				}
				linkVol := v1.Volume{
					Name: "linkVol",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: link,
						},
					},
				}
				pod.Spec.Volumes = append(pod.Spec.Volumes, []v1.Volume{linkVol}...)
				return nil
			},
			shouldFail:    false,
			expectedAdmit: false,
		},
		{
			name: "pod with normal vol",
			prepareVol: func(pod *v1.Pod, base string) error {
				normalDir := path.Join(base, "normal")
				if err := os.Mkdir(normalDir, defaultPerm); err != nil {
					return err
				}
				normalFile := path.Join(base, "normal.txt")

				err := ioutil.WriteFile(normalFile, []byte("hello world"), 0600)
				if err != nil {
					return err
				}
				normalVol := v1.Volume{
					Name: "normalDirVol",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: normalDir,
						},
					},
				}
				normalFileVol := v1.Volume{
					Name: "normalFileVol",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: normalFile,
						},
					},
				}
				pod.Spec.Volumes = append(pod.Spec.Volumes, []v1.Volume{normalVol, normalFileVol}...)
				return nil
			},
			shouldFail:    false,
			expectedAdmit: true,
		},
		{
			name: "pod with notfound vol",
			prepareVol: func(pod *v1.Pod, base string) error {
				notfound := path.Join(base, "notfound")
				notfoundVol := v1.Volume{
					Name: "notfoundVol",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: notfound,
						},
					},
				}
				pod.Spec.Volumes = append(pod.Spec.Volumes, []v1.Volume{notfoundVol}...)
				return nil
			},
			shouldFail:    false,
			expectedAdmit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &v1.Pod{}
			base, err := ioutil.TempDir("", "testsymlink")
			if err != nil {
				fmt.Printf("failed to create tmpdir: %v\n", err)
				return
			}
			defer os.RemoveAll(base)

			if tt.prepareVol == nil {
				t.Fatalf("prepareVol function required")
			}
			tt.prepareVol(pod, base)
			attrs := &PodAdmitAttributes{
				Pod: pod,
			}
			actual := sf.Admit(attrs)
			if tt.shouldFail && err == nil {
				t.Errorf("Expected an error in %v", tt.name)
			} else if !tt.shouldFail && err != nil {
				t.Errorf("Unexpected error in %v, got %v", tt.name, err)
			} else if tt.expectedAdmit != actual.Admit {
				t.Errorf("Failed at case: ['%v'], Expected %v, got %v", tt.name, tt.expectedAdmit, actual.Admit)
			}
		})
	}
}

func TestCheckSymlink(t *testing.T) {
	type returnfield struct {
		err     error
		symlink bool
	}
	tests := []struct {
		name     string
		prepare  func(base string) ([]string, error)
		expected []returnfield
	}{
		{
			name: "check symlink with files",
			prepare: func(base string) ([]string, error) {
				origin := path.Join(base, "original.txt")
				link := path.Join(base, "link.txt")
				notfound := path.Join(base, "notfound")

				err := ioutil.WriteFile(origin, []byte("hello world"), 0600)
				if err != nil {
					return nil, err
				}

				if err = os.Symlink(origin, link); err != nil {
					return nil, err
				}
				return []string{origin, link, notfound}, nil
			},
			expected: []returnfield{
				{nil, false},
				{nil, true},
				{nil, false},
			},
		},
		{
			name: "check symlink with directories",
			prepare: func(base string) ([]string, error) {
				origin := path.Join(base, "original")
				link := path.Join(base, "link")
				notfound := path.Join(base, "notfound")

				if err := os.Mkdir(origin, defaultPerm); err != nil {
					return nil, err
				}

				if err := os.Symlink(origin, link); err != nil {
					return nil, err
				}
				return []string{origin, link, notfound}, nil
			},
			expected: []returnfield{
				{nil, false},
				{nil, true},
				{nil, false},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base, err := ioutil.TempDir("", "testsymlink")
			if err != nil {
				fmt.Printf("failed to create tmpdir: %v\n", err)
				return
			}
			defer os.RemoveAll(base)

			if tt.prepare == nil {
				t.Fatalf("prepare function required")
			}

			paths, err := tt.prepare(base)
			if err != nil {
				t.Fatalf("failed to prepare test: %v", err)
			}
			var symlink bool
			for i, path := range paths {
				expected := tt.expected[i]
				err, symlink = checkSymlink(path)
				if err != expected.err || symlink != expected.symlink {
					t.Fatalf("got: %v, Expected: %v", returnfield{err, symlink}, expected)
				}
			}
		})
	}
}
