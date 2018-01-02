// Copyright 2017 Google Inc. All Rights Reserved.
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

import Bookstore

class Server : Service {
  private var shelves : [Int64:Shelf] = [:]
  private var books : [Int64:[Int64:Book]] = [:]
  private var lastShelfIndex : Int64 = 0
  private var lastBookIndex : Int64 = 0

  // Return all shelves in the bookstore.
  func listShelves () throws -> ListShelvesResponses {
    let responses = ListShelvesResponses()
    let response = ListShelvesResponse()
    var shelves : [Shelf] = []
    for pair in self.shelves {
      shelves.append(pair.value)
    }
    response.shelves = shelves
    responses.ok = response
    return responses
  }
  // Create a new shelf in the bookstore.
  func createShelf (_ parameters : CreateShelfParameters) throws -> CreateShelfResponses {
    lastShelfIndex += 1
    let shelf = parameters.shelf
    shelf.name = "shelves/\(lastShelfIndex)"
    shelves[lastShelfIndex] = shelf
    let responses = CreateShelfResponses()
    responses.ok = shelf
    return responses
  }
  // Delete all shelves.
  func deleteShelves () throws {
    shelves = [:]
    books = [:]
    lastShelfIndex = 0
    lastBookIndex = 0
  }
  // Get a single shelf resource with the given ID.
  func getShelf (_ parameters : GetShelfParameters) throws -> GetShelfResponses {
    let responses =  GetShelfResponses()
    if let shelf : Shelf = shelves[parameters.shelf] {
      responses.ok = shelf
    } else {
      let err = Error()
      err.code = 404
      err.message = "not found"
      responses.error = err
    }
    return responses
  }
  // Delete a single shelf with the given ID.
  func deleteShelf (_ parameters : DeleteShelfParameters) throws {
    shelves[parameters.shelf] = nil
    books[parameters.shelf] = nil
  }
  // Return all books in a shelf with the given ID.
  func listBooks (_ parameters : ListBooksParameters) throws -> ListBooksResponses {
    let responses = ListBooksResponses()
    let response = ListBooksResponse()
    var books : [Book] = []
    if let shelfBooks = self.books[parameters.shelf] {
      for pair in shelfBooks {
        books.append(pair.value)
      }
    }
    response.books = books
    responses.ok = response
    return responses
  }
  // Create a new book on the shelf.
  func createBook (_ parameters : CreateBookParameters) throws -> CreateBookResponses {
    let responses = CreateBookResponses()
    lastBookIndex += 1
    let shelf = parameters.shelf
    let book = parameters.book
    book.name = "shelves/\(shelf)/books/\(lastBookIndex)"
    if var shelfBooks = self.books[shelf] {
      shelfBooks[lastBookIndex] = book
      self.books[shelf] = shelfBooks
    } else {
	  var shelfBooks : [Int64:Book] = [:]
      shelfBooks[lastBookIndex] = book
      self.books[shelf] = shelfBooks
    }
    responses.ok = book
    return responses
  }
  // Get a single book with a given ID from a shelf.
  func getBook (_ parameters : GetBookParameters) throws -> GetBookResponses {
    let responses = GetBookResponses()
    if let shelfBooks = self.books[parameters.shelf],
      let book = shelfBooks[parameters.book] {
      responses.ok = book
    } else {
      let err = Error()
      err.code = 404
      err.message = "not found"
      responses.error = err
    }
    return responses
  }
  // Delete a single book with a given ID from a shelf.
  func deleteBook (_ parameters : DeleteBookParameters) throws {
    if var shelfBooks = self.books[parameters.shelf] {
      shelfBooks[parameters.book] = nil
	  self.books[parameters.shelf] = shelfBooks
    }
  }
}

initialize(service:Server(), port:8080)

run()

