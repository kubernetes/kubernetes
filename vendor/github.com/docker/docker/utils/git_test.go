package utils

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCloneArgsSmartHttp(t *testing.T) {
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	serverURL, _ := url.Parse(server.URL)

	serverURL.Path = "/repo.git"
	gitURL := serverURL.String()

	mux.HandleFunc("/repo.git/info/refs", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("service")
		w.Header().Set("Content-Type", fmt.Sprintf("application/x-%s-advertisement", q))
	})

	args := cloneArgs(serverURL, "/tmp")
	exp := []string{"clone", "--recursive", "--depth", "1", gitURL, "/tmp"}
	if !reflect.DeepEqual(args, exp) {
		t.Fatalf("Expected %v, got %v", exp, args)
	}
}

func TestCloneArgsDumbHttp(t *testing.T) {
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	serverURL, _ := url.Parse(server.URL)

	serverURL.Path = "/repo.git"
	gitURL := serverURL.String()

	mux.HandleFunc("/repo.git/info/refs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
	})

	args := cloneArgs(serverURL, "/tmp")
	exp := []string{"clone", "--recursive", gitURL, "/tmp"}
	if !reflect.DeepEqual(args, exp) {
		t.Fatalf("Expected %v, got %v", exp, args)
	}
}

func TestCloneArgsGit(t *testing.T) {
	u, _ := url.Parse("git://github.com/docker/docker")
	args := cloneArgs(u, "/tmp")
	exp := []string{"clone", "--recursive", "--depth", "1", "git://github.com/docker/docker", "/tmp"}
	if !reflect.DeepEqual(args, exp) {
		t.Fatalf("Expected %v, got %v", exp, args)
	}
}

func TestCloneArgsStripFragment(t *testing.T) {
	u, _ := url.Parse("git://github.com/docker/docker#test")
	args := cloneArgs(u, "/tmp")
	exp := []string{"clone", "--recursive", "git://github.com/docker/docker", "/tmp"}
	if !reflect.DeepEqual(args, exp) {
		t.Fatalf("Expected %v, got %v", exp, args)
	}
}

func TestCheckoutGit(t *testing.T) {
	root, err := ioutil.TempDir("", "docker-build-git-checkout")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)

	gitDir := filepath.Join(root, "repo")
	_, err = git("init", gitDir)
	if err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "config", "user.email", "test@docker.com"); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "config", "user.name", "Docker test"); err != nil {
		t.Fatal(err)
	}

	if err = ioutil.WriteFile(filepath.Join(gitDir, "Dockerfile"), []byte("FROM scratch"), 0644); err != nil {
		t.Fatal(err)
	}

	subDir := filepath.Join(gitDir, "subdir")
	if err = os.Mkdir(subDir, 0755); err != nil {
		t.Fatal(err)
	}

	if err = ioutil.WriteFile(filepath.Join(subDir, "Dockerfile"), []byte("FROM scratch\nEXPOSE 5000"), 0644); err != nil {
		t.Fatal(err)
	}

	if err = os.Symlink("../subdir", filepath.Join(gitDir, "parentlink")); err != nil {
		t.Fatal(err)
	}

	if err = os.Symlink("/subdir", filepath.Join(gitDir, "absolutelink")); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "add", "-A"); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "commit", "-am", "First commit"); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "checkout", "-b", "test"); err != nil {
		t.Fatal(err)
	}

	if err = ioutil.WriteFile(filepath.Join(gitDir, "Dockerfile"), []byte("FROM scratch\nEXPOSE 3000"), 0644); err != nil {
		t.Fatal(err)
	}

	if err = ioutil.WriteFile(filepath.Join(subDir, "Dockerfile"), []byte("FROM busybox\nEXPOSE 5000"), 0644); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "add", "-A"); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "commit", "-am", "Branch commit"); err != nil {
		t.Fatal(err)
	}

	if _, err = gitWithinDir(gitDir, "checkout", "master"); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		frag string
		exp  string
		fail bool
	}{
		{"", "FROM scratch", false},
		{"master", "FROM scratch", false},
		{":subdir", "FROM scratch\nEXPOSE 5000", false},
		{":nosubdir", "", true},   // missing directory error
		{":Dockerfile", "", true}, // not a directory error
		{"master:nosubdir", "", true},
		{"master:subdir", "FROM scratch\nEXPOSE 5000", false},
		{"master:parentlink", "FROM scratch\nEXPOSE 5000", false},
		{"master:absolutelink", "FROM scratch\nEXPOSE 5000", false},
		{"master:../subdir", "", true},
		{"test", "FROM scratch\nEXPOSE 3000", false},
		{"test:", "FROM scratch\nEXPOSE 3000", false},
		{"test:subdir", "FROM busybox\nEXPOSE 5000", false},
	}

	for _, c := range cases {
		r, err := checkoutGit(c.frag, gitDir)

		fail := err != nil
		if fail != c.fail {
			t.Fatalf("Expected %v failure, error was %v\n", c.fail, err)
		}
		if c.fail {
			continue
		}

		b, err := ioutil.ReadFile(filepath.Join(r, "Dockerfile"))
		if err != nil {
			t.Fatal(err)
		}

		if string(b) != c.exp {
			t.Fatalf("Expected %v, was %v\n", c.exp, string(b))
		}
	}
}
