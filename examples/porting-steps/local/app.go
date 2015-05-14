/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"database/sql"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

func connect() (*sql.DB, error) {
	db, err := sql.Open("mysql", "root:secret@tcp(localhost:3306)/?parseTime=true")
	if err != nil {
		return db, fmt.Errorf("Error opening db: %v", err)
	}

	_, err = db.Exec(
		"CREATE DATABASE IF NOT EXISTS guestbook;")
	if err != nil {
		return db, fmt.Errorf("Error creating db: %v", err)
	}

	_, err = db.Exec(
		"USE guestbook;")
	if err != nil {
		return db, fmt.Errorf("Error using db: %v", err)
	}

	_, err = db.Exec(
		"CREATE TABLE IF NOT EXISTS entries " +
			"(date DATETIME PRIMARY KEY, entry VARCHAR(256));")
	if err != nil {
		return db, fmt.Errorf("Error creating table: %v", err)
	}

	log.Printf("Database connected and setup")
	return db, nil
}

type Post struct {
	Date    time.Time
	Content string
}

func getPosts(db *sql.DB) ([]Post, error) {
	rows, err := db.Query("SELECT date, entry FROM entries;")
	if err != nil {
		return nil, fmt.Errorf("Error getting users tweets: %v", err)
	}
	posts := make([]Post, 0)
	for rows.Next() {
		var date time.Time
		var content string
		err = rows.Scan(&date, &content)
		if err != nil {
			return nil, fmt.Errorf("Couldn't get posts from db: %v", err)
		}
		posts = append(posts, Post{Date: date, Content: content})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("Row error: %v", err)
	}
	return posts, nil
}

func addPost(db *sql.DB, pp Post) error {
	_, err := db.Exec(
		"INSERT INTO entries (date, entry)"+
			"VALUES (?, ?);",
		pp.Date,
		pp.Content,
	)
	if err != nil {
		return fmt.Errorf("Error inserting post: %v", err)
	}
	return nil
}

type GBHandler struct {
	db *sql.DB
	ts *template.Template
}

func (hh GBHandler) ServeHTTP(ww http.ResponseWriter, rr *http.Request) {
	var info struct {
		Posts []Post
	}
	var err error
	info.Posts, err = getPosts(hh.db)
	if err != nil {
		http.Error(ww, err.Error(), http.StatusInternalServerError)
	}
	if err := hh.ts.ExecuteTemplate(
		ww,
		"main.html",
		info); err != nil {
		http.Error(ww, err.Error(), http.StatusInternalServerError)
	}
}

type PHandler struct {
	db *sql.DB
}

func (hh PHandler) ServeHTTP(ww http.ResponseWriter, rr *http.Request) {
	pp := Post{Date: time.Now(), Content: rr.FormValue("post")}
	if err := addPost(hh.db, pp); err != nil {
		http.Error(ww, err.Error(), http.StatusInternalServerError)
	}
	http.Redirect(ww, rr, "/", http.StatusFound)
}

func main() {
	db, err := connect()
	if err != nil {
		log.Fatal(err.Error())
	}

	ts := template.Must(template.ParseFiles("main.html"))

	gbh := GBHandler{db: db, ts: ts}
	http.Handle("/", gbh)

	ph := PHandler{db: db}
	http.Handle("/add", ph)

	http.ListenAndServe(":8080", nil)
}
