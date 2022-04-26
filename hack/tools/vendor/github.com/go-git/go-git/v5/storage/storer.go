package storage

import (
	"errors"

	"github.com/go-git/go-git/v5/config"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

var ErrReferenceHasChanged = errors.New("reference has changed concurrently")

// Storer is a generic storage of objects, references and any information
// related to a particular repository. The package github.com/go-git/go-git/v5/storage
// contains two implementation a filesystem base implementation (such as `.git`)
// and a memory implementations being ephemeral
type Storer interface {
	storer.EncodedObjectStorer
	storer.ReferenceStorer
	storer.ShallowStorer
	storer.IndexStorer
	config.ConfigStorer
	ModuleStorer
}

// ModuleStorer allows interact with the modules' Storers
type ModuleStorer interface {
	// Module returns a Storer representing a submodule, if not exists returns a
	// new empty Storer is returned
	Module(name string) (Storer, error)
}
