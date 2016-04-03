package graph

import (
	"errors"
	"io"
	"io/ioutil"
	"os"
	"path"
	"testing"
	"time"

	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/stringid"
)

func TestMount(t *testing.T) {
	graph, driver := tempGraph(t)
	defer os.RemoveAll(graph.root)
	defer driver.Cleanup()

	archive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	image, err := graph.Create(archive, "", "", "Testing", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	tmp, err := ioutil.TempDir("", "docker-test-graph-mount-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	rootfs := path.Join(tmp, "rootfs")
	if err := os.MkdirAll(rootfs, 0700); err != nil {
		t.Fatal(err)
	}
	rw := path.Join(tmp, "rw")
	if err := os.MkdirAll(rw, 0700); err != nil {
		t.Fatal(err)
	}

	if _, err := driver.Get(image.ID, ""); err != nil {
		t.Fatal(err)
	}

}

func TestInit(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	// Root should exist
	if _, err := os.Stat(graph.root); err != nil {
		t.Fatal(err)
	}
	// Map() should be empty
	l := graph.Map()
	if len(l) != 0 {
		t.Fatalf("len(Map()) should return %d, not %d", 0, len(l))
	}
}

// Test that Register can be interrupted cleanly without side effects
func TestInterruptedRegister(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	badArchive, w := io.Pipe() // Use a pipe reader as a fake archive which never yields data
	image := &image.Image{
		ID:      stringid.GenerateRandomID(),
		Comment: "testing",
		Created: time.Now(),
	}
	w.CloseWithError(errors.New("But I'm not a tarball!")) // (Nobody's perfect, darling)
	graph.Register(image, badArchive)
	if _, err := graph.Get(image.ID); err == nil {
		t.Fatal("Image should not exist after Register is interrupted")
	}
	// Registering the same image again should succeed if the first register was interrupted
	goodArchive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	if err := graph.Register(image, goodArchive); err != nil {
		t.Fatal(err)
	}
}

// FIXME: Do more extensive tests (ex: create multiple, delete, recreate;
//       create multiple, check the amount of images and paths, etc..)
func TestGraphCreate(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	archive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	img, err := graph.Create(archive, "", "", "Testing", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := image.ValidateID(img.ID); err != nil {
		t.Fatal(err)
	}
	if img.Comment != "Testing" {
		t.Fatalf("Wrong comment: should be '%s', not '%s'", "Testing", img.Comment)
	}
	if img.DockerVersion != dockerversion.VERSION {
		t.Fatalf("Wrong docker_version: should be '%s', not '%s'", dockerversion.VERSION, img.DockerVersion)
	}
	images := graph.Map()
	if l := len(images); l != 1 {
		t.Fatalf("Wrong number of images. Should be %d, not %d", 1, l)
	}
	if images[img.ID] == nil {
		t.Fatalf("Could not find image with id %s", img.ID)
	}
}

func TestRegister(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	archive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	image := &image.Image{
		ID:      stringid.GenerateRandomID(),
		Comment: "testing",
		Created: time.Now(),
	}
	err = graph.Register(image, archive)
	if err != nil {
		t.Fatal(err)
	}
	images := graph.Map()
	if l := len(images); l != 1 {
		t.Fatalf("Wrong number of images. Should be %d, not %d", 1, l)
	}
	if resultImg, err := graph.Get(image.ID); err != nil {
		t.Fatal(err)
	} else {
		if resultImg.ID != image.ID {
			t.Fatalf("Wrong image ID. Should be '%s', not '%s'", image.ID, resultImg.ID)
		}
		if resultImg.Comment != image.Comment {
			t.Fatalf("Wrong image comment. Should be '%s', not '%s'", image.Comment, resultImg.Comment)
		}
	}
}

// Test that an image can be deleted by its shorthand prefix
func TestDeletePrefix(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	img := createTestImage(graph, t)
	if err := graph.Delete(stringid.TruncateID(img.ID)); err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 0)
}

func TestDelete(t *testing.T) {
	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	archive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 0)
	img, err := graph.Create(archive, "", "", "Bla bla", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 1)
	if err := graph.Delete(img.ID); err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 0)

	archive, err = fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	// Test 2 create (same name) / 1 delete
	img1, err := graph.Create(archive, "", "", "Testing", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	archive, err = fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = graph.Create(archive, "", "", "Testing", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 2)
	if err := graph.Delete(img1.ID); err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 1)

	// Test delete wrong name
	if err := graph.Delete("Not_foo"); err == nil {
		t.Fatalf("Deleting wrong ID should return an error")
	}
	assertNImages(graph, t, 1)

	archive, err = fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	// Test delete twice (pull -> rm -> pull -> rm)
	if err := graph.Register(img1, archive); err != nil {
		t.Fatal(err)
	}
	if err := graph.Delete(img1.ID); err != nil {
		t.Fatal(err)
	}
	assertNImages(graph, t, 1)
}

func TestByParent(t *testing.T) {
	archive1, _ := fakeTar()
	archive2, _ := fakeTar()
	archive3, _ := fakeTar()

	graph, _ := tempGraph(t)
	defer nukeGraph(graph)
	parentImage := &image.Image{
		ID:      stringid.GenerateRandomID(),
		Comment: "parent",
		Created: time.Now(),
		Parent:  "",
	}
	childImage1 := &image.Image{
		ID:      stringid.GenerateRandomID(),
		Comment: "child1",
		Created: time.Now(),
		Parent:  parentImage.ID,
	}
	childImage2 := &image.Image{
		ID:      stringid.GenerateRandomID(),
		Comment: "child2",
		Created: time.Now(),
		Parent:  parentImage.ID,
	}
	_ = graph.Register(parentImage, archive1)
	_ = graph.Register(childImage1, archive2)
	_ = graph.Register(childImage2, archive3)

	byParent := graph.ByParent()
	numChildren := len(byParent[parentImage.ID])
	if numChildren != 2 {
		t.Fatalf("Expected 2 children, found %d", numChildren)
	}
}

func createTestImage(graph *Graph, t *testing.T) *image.Image {
	archive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	img, err := graph.Create(archive, "", "", "Test image", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func assertNImages(graph *Graph, t *testing.T, n int) {
	images := graph.Map()
	if actualN := len(images); actualN != n {
		t.Fatalf("Expected %d images, found %d", n, actualN)
	}
}

func tempGraph(t *testing.T) (*Graph, graphdriver.Driver) {
	tmp, err := ioutil.TempDir("", "docker-graph-")
	if err != nil {
		t.Fatal(err)
	}
	driver, err := graphdriver.New(tmp, nil)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewGraph(tmp, driver)
	if err != nil {
		t.Fatal(err)
	}
	return graph, driver
}

func nukeGraph(graph *Graph) {
	graph.driver.Cleanup()
	os.RemoveAll(graph.root)
}
