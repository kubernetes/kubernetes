package gitserver

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/euforia/go-git-server/repository"
)

type RepoHTTPService struct {
	repos repository.RepositoryStore
}

func NewRepoHTTPService(store repository.RepositoryStore) *RepoHTTPService {
	return &RepoHTTPService{repos: store}
}

func (svr *RepoHTTPService) ServeHTTP(w http.ResponseWriter, r *http.Request) {

	ctx := r.Context()
	repoID := ctx.Value("ID").(string)

	if !strings.Contains(repoID, "/") {
		w.WriteHeader(404)
		return
	}

	var (
		err  error
		repo *repository.Repository
	)

	switch r.Method {
	case "GET":
		repo, err = svr.repos.GetRepo(repoID)

	case "PUT":
		// Create
		dec := json.NewDecoder(r.Body)
		defer r.Body.Close()

		repo = repository.NewRepository(repoID)
		if err = dec.Decode(&repo); err == nil || err == io.EOF {
			err = svr.repos.CreateRepo(repo)
		}

	case "POST":
		// Update
		dec := json.NewDecoder(r.Body)
		defer r.Body.Close()

		// Get existing
		if repo, err = svr.repos.GetRepo(repoID); err == nil {
			// Unmarshal on to existing
			if err = dec.Decode(repo); err == nil {
				err = svr.repos.UpdateRepo(repo)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")

	if err != nil {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":"` + err.Error() + `"}`))
	} else {
		b, _ := json.Marshal(repo)
		w.WriteHeader(200)
		w.Write(b)
	}

}
