/*
 Copyright 2017 Google Inc. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package main

import (
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/googleapis/gnostic/plugins/gnostic-go-generator/examples/v2.0/bookstore/bookstore"
)

//
// The Service type implements a bookstore service.
// All objects are managed in an in-memory non-persistent store.
//
type Service struct {
	// shelves are stored in a map keyed by shelf id
	// books are stored in a two level map, keyed first by shelf id and then by book id
	Shelves     map[int64]*bookstore.Shelf
	Books       map[int64]map[int64]*bookstore.Book
	LastShelfID int64      // the id of the last shelf that was added
	LastBookID  int64      // the id of the last book that was added
	Mutex       sync.Mutex // global mutex to synchronize service access
}

func NewService() *Service {
	return &Service{
		Shelves: make(map[int64]*bookstore.Shelf),
		Books:   make(map[int64]map[int64]*bookstore.Book),
	}
}

func (service *Service) ListShelves(responses *bookstore.ListShelvesResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// copy shelf ids from Shelves map keys
	shelves := make([]bookstore.Shelf, 0, len(service.Shelves))
	for _, shelf := range service.Shelves {
		shelves = append(shelves, *shelf)
	}
	response := &bookstore.ListShelvesResponse{}
	response.Shelves = shelves
	(*responses).OK = response
	return err
}

func (service *Service) CreateShelf(parameters *bookstore.CreateShelfParameters, responses *bookstore.CreateShelfResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// assign an id and name to a shelf and add it to the Shelves map.
	shelf := parameters.Shelf
	service.LastShelfID++
	sid := service.LastShelfID
	shelf.Name = fmt.Sprintf("shelves/%d", sid)
	service.Shelves[sid] = &shelf
	(*responses).OK = &shelf
	return err
}

func (service *Service) DeleteShelves() (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// delete everything by reinitializing the Shelves and Books maps.
	service.Shelves = make(map[int64]*bookstore.Shelf)
	service.Books = make(map[int64]map[int64]*bookstore.Book)
	service.LastShelfID = 0
	service.LastBookID = 0
	return nil
}

func (service *Service) GetShelf(parameters *bookstore.GetShelfParameters, responses *bookstore.GetShelfResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// look up a shelf from the Shelves map.
	shelf, err := service.getShelf(parameters.Shelf)
	if err != nil {
		(*responses).Default = &bookstore.Error{Code: int32(http.StatusNotFound), Message: err.Error()}
		return nil
	} else {
		(*responses).OK = shelf
		return nil
	}
}

func (service *Service) DeleteShelf(parameters *bookstore.DeleteShelfParameters) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// delete a shelf by removing the shelf from the Shelves map and the associated books from the Books map.
	delete(service.Shelves, parameters.Shelf)
	delete(service.Books, parameters.Shelf)
	return nil
}

func (service *Service) ListBooks(parameters *bookstore.ListBooksParameters, responses *bookstore.ListBooksResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// list the books in a shelf
	_, err = service.getShelf(parameters.Shelf)
	if err != nil {
		(*responses).Default = &bookstore.Error{Code: int32(http.StatusNotFound), Message: err.Error()}
		return nil
	}
	shelfBooks := service.Books[parameters.Shelf]
	books := make([]bookstore.Book, 0, len(shelfBooks))
	for _, book := range shelfBooks {
		books = append(books, *book)
	}
	response := &bookstore.ListBooksResponse{}
	response.Books = books
	(*responses).OK = response
	return nil
}

func (service *Service) CreateBook(parameters *bookstore.CreateBookParameters, responses *bookstore.CreateBookResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// return "not found" if the shelf doesn't exist
	shelf, err := service.getShelf(parameters.Shelf)
	if err != nil {
		(*responses).Default = &bookstore.Error{Code: int32(http.StatusNotFound), Message: err.Error()}
		return nil
	}
	// assign an id and name to a book and add it to the Books map.
	service.LastBookID++
	bid := service.LastBookID
	book := parameters.Book
	book.Name = fmt.Sprintf("%s/books/%d", shelf.Name, bid)
	if service.Books[parameters.Shelf] == nil {
		service.Books[parameters.Shelf] = make(map[int64]*bookstore.Book)
	}
	service.Books[parameters.Shelf][bid] = &book
	(*responses).OK = &book
	return err
}

func (service *Service) GetBook(parameters *bookstore.GetBookParameters, responses *bookstore.GetBookResponses) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// get a book from the Books map
	book, err := service.getBook(parameters.Shelf, parameters.Book)
	if err != nil {
		(*responses).Default = &bookstore.Error{Code: int32(http.StatusNotFound), Message: err.Error()}
	} else {
		(*responses).OK = book
	}
	return nil
}

func (service *Service) DeleteBook(parameters *bookstore.DeleteBookParameters) (err error) {
	service.Mutex.Lock()
	defer service.Mutex.Unlock()
	// delete a book by removing the book from the Books map.
	delete(service.Books[parameters.Shelf], parameters.Book)
	return nil
}

// internal helpers

func (service *Service) getShelf(sid int64) (shelf *bookstore.Shelf, err error) {
	shelf, ok := service.Shelves[sid]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Couldn't find shelf %d", sid))
	} else {
		return shelf, nil
	}
}

func (service *Service) getBook(sid int64, bid int64) (book *bookstore.Book, err error) {
	_, err = service.getShelf(sid)
	if err != nil {
		return nil, err
	}
	book, ok := service.Books[sid][bid]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Couldn't find book %d on shelf %d", bid, sid))
	} else {
		return book, nil
	}
}
