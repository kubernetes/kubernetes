package repository

// Repository represents a single repo.
type Repository struct {
	ID   string                `json:"id"`
	Refs *RepositoryReferences `json:"refs"`
}

// NewRepository instantiates an empty repo.
func NewRepository(id string) *Repository {
	return &Repository{ID: id, Refs: NewRepositoryReferences()}
}

func (repo *Repository) String() string {
	return repo.ID
}
