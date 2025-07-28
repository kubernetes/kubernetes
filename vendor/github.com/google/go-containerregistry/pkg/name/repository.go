// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package name

import (
	"encoding"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	defaultNamespace = "library"
	repositoryChars  = "abcdefghijklmnopqrstuvwxyz0123456789_-./"
	regRepoDelimiter = "/"
)

// Repository stores a docker repository name in a structured form.
type Repository struct {
	Registry
	repository string
}

var _ encoding.TextMarshaler = (*Repository)(nil)
var _ encoding.TextUnmarshaler = (*Repository)(nil)
var _ json.Marshaler = (*Repository)(nil)
var _ json.Unmarshaler = (*Repository)(nil)

// See https://docs.docker.com/docker-hub/official_repos
func hasImplicitNamespace(repo string, reg Registry) bool {
	return !strings.ContainsRune(repo, '/') && reg.RegistryStr() == DefaultRegistry
}

// RepositoryStr returns the repository component of the Repository.
func (r Repository) RepositoryStr() string {
	if hasImplicitNamespace(r.repository, r.Registry) {
		return fmt.Sprintf("%s/%s", defaultNamespace, r.repository)
	}
	return r.repository
}

// Name returns the name from which the Repository was derived.
func (r Repository) Name() string {
	regName := r.Registry.Name()
	if regName != "" {
		return regName + regRepoDelimiter + r.RepositoryStr()
	}
	// TODO: As far as I can tell, this is unreachable.
	return r.RepositoryStr()
}

func (r Repository) String() string {
	return r.Name()
}

// Scope returns the scope required to perform the given action on the registry.
// TODO(jonjohnsonjr): consider moving scopes to a separate package.
func (r Repository) Scope(action string) string {
	return fmt.Sprintf("repository:%s:%s", r.RepositoryStr(), action)
}

func checkRepository(repository string) error {
	return checkElement("repository", repository, repositoryChars, 2, 255)
}

// NewRepository returns a new Repository representing the given name, according to the given strictness.
func NewRepository(name string, opts ...Option) (Repository, error) {
	opt := makeOptions(opts...)
	if len(name) == 0 {
		return Repository{}, newErrBadName("a repository name must be specified")
	}

	var registry string
	repo := name
	parts := strings.SplitN(name, regRepoDelimiter, 2)
	if len(parts) == 2 && (strings.ContainsRune(parts[0], '.') || strings.ContainsRune(parts[0], ':')) {
		// The first part of the repository is treated as the registry domain
		// iff it contains a '.' or ':' character, otherwise it is all repository
		// and the domain defaults to Docker Hub.
		registry = parts[0]
		repo = parts[1]
	}

	if err := checkRepository(repo); err != nil {
		return Repository{}, err
	}

	reg, err := NewRegistry(registry, opts...)
	if err != nil {
		return Repository{}, err
	}
	if hasImplicitNamespace(repo, reg) && opt.strict {
		return Repository{}, newErrBadName("strict validation requires the full repository path (missing 'library')")
	}
	return Repository{reg, repo}, nil
}

// Tag returns a Tag in this Repository.
func (r Repository) Tag(identifier string) Tag {
	t := Tag{
		tag:        identifier,
		Repository: r,
	}
	t.original = t.Name()
	return t
}

// Digest returns a Digest in this Repository.
func (r Repository) Digest(identifier string) Digest {
	d := Digest{
		digest:     identifier,
		Repository: r,
	}
	d.original = d.Name()
	return d
}

// MarshalJSON formats the Repository into a string for JSON serialization.
func (r Repository) MarshalJSON() ([]byte, error) { return json.Marshal(r.String()) }

// UnmarshalJSON parses a JSON string into a Repository.
func (r *Repository) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	n, err := NewRepository(s)
	if err != nil {
		return err
	}
	*r = n
	return nil
}

// MarshalText formats the repository name into a string for text serialization.
func (r Repository) MarshalText() ([]byte, error) { return []byte(r.String()), nil }

// UnmarshalText parses a text string into a Repository.
func (r *Repository) UnmarshalText(data []byte) error {
	n, err := NewRepository(string(data))
	if err != nil {
		return err
	}
	*r = n
	return nil
}
