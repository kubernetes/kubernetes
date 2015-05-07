package github

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
)

func TestDecode(t *testing.T) {
	setup()
	defer teardown()
	r := RepositoryContent{Encoding: String("base64"), Content: String("aGVsbG8=")}
	o, err := r.Decode()
	if err != nil {
		t.Errorf("Failed to decode content.")
	}
	want := "hello"
	if string(o) != want {
		t.Errorf("RepositoryContent.Decode returned %+v, want %+v", string(o), want)
	}
}

func TestDecodeBadEncoding(t *testing.T) {
	setup()
	defer teardown()
	r := RepositoryContent{Encoding: String("bad")}
	_, err := r.Decode()
	if err == nil {
		t.Errorf("Should fail to decode non-base64")
	}
}

func TestRepositoriesService_GetReadme(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/readme", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{
		  "type": "file",
		  "encoding": "base64",
		  "size": 5362,
		  "name": "README.md",
		  "path": "README.md"
		}`)
	})
	readme, _, err := client.Repositories.GetReadme("o", "r", &RepositoryContentGetOptions{})
	if err != nil {
		t.Errorf("Repositories.GetReadme returned error: %v", err)
	}
	want := &RepositoryContent{Type: String("file"), Name: String("README.md"), Size: Int(5362), Encoding: String("base64"), Path: String("README.md")}
	if !reflect.DeepEqual(readme, want) {
		t.Errorf("Repositories.GetReadme returned %+v, want %+v", readme, want)
	}
}

func TestRepositoriesService_DownloadContents_Success(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/d", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{
		  "type": "file",
		  "name": "f",
		  "download_url": "`+server.URL+`/download/f"
		}]`)
	})
	mux.HandleFunc("/download/f", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, "foo")
	})

	r, err := client.Repositories.DownloadContents("o", "r", "d/f", nil)
	if err != nil {
		t.Errorf("Repositories.DownloadContents returned error: %v", err)
	}

	bytes, err := ioutil.ReadAll(r)
	if err != nil {
		t.Errorf("Error reading response body: %v", err)
	}
	r.Close()

	if got, want := string(bytes), "foo"; got != want {
		t.Errorf("Repositories.DownloadContents returned %v, want %v", got, want)
	}
}

func TestRepositoriesService_DownloadContents_NoDownloadURL(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/d", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{
		  "type": "file",
		  "name": "f",
		}]`)
	})

	_, err := client.Repositories.DownloadContents("o", "r", "d/f", nil)
	if err == nil {
		t.Errorf("Repositories.DownloadContents did not return expected error")
	}
}

func TestRepositoriesService_DownloadContents_NoFile(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/d", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[]`)
	})

	_, err := client.Repositories.DownloadContents("o", "r", "d/f", nil)
	if err == nil {
		t.Errorf("Repositories.DownloadContents did not return expected error")
	}
}

func TestRepositoriesService_GetContents_File(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/p", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{
		  "type": "file",
		  "encoding": "base64",
		  "size": 20678,
		  "name": "LICENSE",
		  "path": "LICENSE"
		}`)
	})
	fileContents, _, _, err := client.Repositories.GetContents("o", "r", "p", &RepositoryContentGetOptions{})
	if err != nil {
		t.Errorf("Repositories.GetContents returned error: %v", err)
	}
	want := &RepositoryContent{Type: String("file"), Name: String("LICENSE"), Size: Int(20678), Encoding: String("base64"), Path: String("LICENSE")}
	if !reflect.DeepEqual(fileContents, want) {
		t.Errorf("Repositories.GetContents returned %+v, want %+v", fileContents, want)
	}
}

func TestRepositoriesService_GetContents_Directory(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/p", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{
		  "type": "dir",
		  "name": "lib",
		  "path": "lib"
		},
		{
		  "type": "file",
		  "size": 20678,
		  "name": "LICENSE",
		  "path": "LICENSE"
		}]`)
	})
	_, directoryContents, _, err := client.Repositories.GetContents("o", "r", "p", &RepositoryContentGetOptions{})
	if err != nil {
		t.Errorf("Repositories.GetContents returned error: %v", err)
	}
	want := []*RepositoryContent{{Type: String("dir"), Name: String("lib"), Path: String("lib")},
		{Type: String("file"), Name: String("LICENSE"), Size: Int(20678), Path: String("LICENSE")}}
	if !reflect.DeepEqual(directoryContents, want) {
		t.Errorf("Repositories.GetContents_Directory returned %+v, want %+v", directoryContents, want)
	}
}

func TestRepositoriesService_CreateFile(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/p", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		fmt.Fprint(w, `{
			"content":{
				"name":"p"
			},
			"commit":{
				"message":"m",
				"sha":"f5f369044773ff9c6383c087466d12adb6fa0828"
			}
		}`)
	})
	message := "m"
	content := []byte("c")
	repositoryContentsOptions := &RepositoryContentFileOptions{
		Message:   &message,
		Content:   content,
		Committer: &CommitAuthor{Name: String("n"), Email: String("e")},
	}
	createResponse, _, err := client.Repositories.CreateFile("o", "r", "p", repositoryContentsOptions)
	if err != nil {
		t.Errorf("Repositories.CreateFile returned error: %v", err)
	}
	want := &RepositoryContentResponse{
		Content: &RepositoryContent{Name: String("p")},
		Commit: Commit{
			Message: String("m"),
			SHA:     String("f5f369044773ff9c6383c087466d12adb6fa0828"),
		},
	}
	if !reflect.DeepEqual(createResponse, want) {
		t.Errorf("Repositories.CreateFile returned %+v, want %+v", createResponse, want)
	}
}

func TestRepositoriesService_UpdateFile(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/p", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		fmt.Fprint(w, `{
			"content":{
				"name":"p"
			},
			"commit":{
				"message":"m",
				"sha":"f5f369044773ff9c6383c087466d12adb6fa0828"
			}
		}`)
	})
	message := "m"
	content := []byte("c")
	sha := "f5f369044773ff9c6383c087466d12adb6fa0828"
	repositoryContentsOptions := &RepositoryContentFileOptions{
		Message:   &message,
		Content:   content,
		SHA:       &sha,
		Committer: &CommitAuthor{Name: String("n"), Email: String("e")},
	}
	updateResponse, _, err := client.Repositories.UpdateFile("o", "r", "p", repositoryContentsOptions)
	if err != nil {
		t.Errorf("Repositories.UpdateFile returned error: %v", err)
	}
	want := &RepositoryContentResponse{
		Content: &RepositoryContent{Name: String("p")},
		Commit: Commit{
			Message: String("m"),
			SHA:     String("f5f369044773ff9c6383c087466d12adb6fa0828"),
		},
	}
	if !reflect.DeepEqual(updateResponse, want) {
		t.Errorf("Repositories.UpdateFile returned %+v, want %+v", updateResponse, want)
	}
}

func TestRepositoriesService_DeleteFile(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/contents/p", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		fmt.Fprint(w, `{
			"content": null,
			"commit":{
				"message":"m",
				"sha":"f5f369044773ff9c6383c087466d12adb6fa0828"
			}
		}`)
	})
	message := "m"
	sha := "f5f369044773ff9c6383c087466d12adb6fa0828"
	repositoryContentsOptions := &RepositoryContentFileOptions{
		Message:   &message,
		SHA:       &sha,
		Committer: &CommitAuthor{Name: String("n"), Email: String("e")},
	}
	deleteResponse, _, err := client.Repositories.DeleteFile("o", "r", "p", repositoryContentsOptions)
	if err != nil {
		t.Errorf("Repositories.DeleteFile returned error: %v", err)
	}
	want := &RepositoryContentResponse{
		Content: nil,
		Commit: Commit{
			Message: String("m"),
			SHA:     String("f5f369044773ff9c6383c087466d12adb6fa0828"),
		},
	}
	if !reflect.DeepEqual(deleteResponse, want) {
		t.Errorf("Repositories.DeleteFile returned %+v, want %+v", deleteResponse, want)
	}
}

func TestRepositoriesService_GetArchiveLink(t *testing.T) {
	setup()
	defer teardown()
	mux.HandleFunc("/repos/o/r/tarball", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Redirect(w, r, "http://github.com/a", http.StatusFound)
	})
	url, resp, err := client.Repositories.GetArchiveLink("o", "r", Tarball, &RepositoryContentGetOptions{})
	if err != nil {
		t.Errorf("Repositories.GetArchiveLink returned error: %v", err)
	}
	if resp.StatusCode != http.StatusFound {
		t.Errorf("Repositories.GetArchiveLink returned status: %d, want %d", resp.StatusCode, http.StatusFound)
	}
	want := "http://github.com/a"
	if url.String() != want {
		t.Errorf("Repositories.GetArchiveLink returned %+v, want %+v", url.String(), want)
	}
}
