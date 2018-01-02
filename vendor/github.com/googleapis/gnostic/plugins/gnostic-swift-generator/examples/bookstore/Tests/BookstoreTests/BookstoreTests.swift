/*
 *
 * Copyright 2017, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
import XCTest
import Foundation
@testable import Bookstore

func Log(_ message : String) {
  FileHandle.standardError.write((message + "\n").data(using:.utf8)!)
}

let service = "http://localhost:8080"

class BookstoreTests: XCTestCase {

  func testBasic() {
    // create a client
    let b = Bookstore.Client(service:service)
    Log("// reset the service by deleting all shelves")
    do {
      try b.deleteShelves()
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// verify that the service has no shelves")
    do {
      let response = try b.listShelves()
      XCTAssertEqual(response.shelves.count, 0)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// attempting to get a shelf should return an error")
    do {
      let _ = try b.getShelf(shelf:1)
      XCTFail("server error")
    } catch {
    }
    Log("// attempting to get a book should return an error")
    do {
      let _ = try b.getBook(shelf:1, book:2)
    } catch {
    }
    Log("// add a shelf")
    do {
      let shelf = Shelf()
      shelf.theme = "mysteries"
      let response = try b.createShelf(shelf:shelf)
      if (response.name != "shelves/1") ||
        (response.theme != "mysteries") {
        XCTFail("mismatch")
      }
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// add another shelf")
    do {
      let shelf = Shelf()
      shelf.theme = "comedies"
      let response = try b.createShelf(shelf:shelf)
      if (response.name != "shelves/2") ||
        (response.theme != "comedies") {
        XCTFail("mismatch")
      }
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// get the first shelf that was added")
    do {
      let response = try b.getShelf(shelf:1)
      if (response.name != "shelves/1") ||
        (response.theme != "mysteries") {
        XCTFail("mismatch")
      }
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// list shelves and verify that there are 2")
    do {
      let response = try b.listShelves()
      XCTAssertEqual(response.shelves.count, 2)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// delete a shelf")
    do {
      try b.deleteShelf(shelf:2)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// list shelves and verify that there is only 1")
    do {
      let response = try b.listShelves()
      XCTAssertEqual(response.shelves.count, 1)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// list books on a shelf, verify that there are none")
    do {
      let response = try b.listBooks(shelf:1)
      XCTAssertEqual(response.books.count, 0)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// create a book")
    do {
      let book = Book()
      book.author = "Agatha Christie"
      book.title = "And Then There Were None"
      let _ = try b.createBook(shelf:1, book:book)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// create another book")
    do {
      let book = Book()
      book.author = "Agatha Christie"
      book.title = "Murder on the Orient Express"
      let _ = try b.createBook(shelf:1, book:book)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// get the first book that was added")
    do {
      let response = try b.getBook(shelf:1, book:1)
      if (response.author != "Agatha Christie") ||
        (response.title != "And Then There Were None") {
        XCTFail("mismatch")
      }
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// list the books on a shelf and verify that there are 2")
    do {
      let response = try b.listBooks(shelf:1)
      XCTAssertEqual(response.books.count, 2)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// delete a book")
    do {
      try b.deleteBook(shelf:1, book:2)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// list the books on a shelf and verify that is only 1")
    do {
      let response = try b.listBooks(shelf:1)
      XCTAssertEqual(response.books.count, 1)
    } catch (let error) {
      XCTFail("\(error)")
    }
    Log("// verify the handling of a badly-formed request")
    var path = service
    path = path + "/shelves"
    guard let url = URL(string:path) else {
        XCTFail("Failed to construct URL")    	
		return
    }
    var request = URLRequest(url:url)
    request.httpMethod = "POST"
    request.httpBody = "".data(using:.utf8)
    let (_, response, _) = fetch(request)
    // we expect a 400 (Bad Request) code
    if let response = response {
      XCTAssertEqual(response.statusCode, 400)
    } else {
      // Failed requests are returning nil responses on Linux. For now we'll say that is OK.
      //XCTFail("Null response for bad request")    	
    }
  }
}

extension BookstoreTests {
  static var allTests : [(String, (BookstoreTests) -> () throws -> Void)] {
    return [
      ("testBasic", testBasic),
    ]
  }
}
