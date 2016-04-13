package remote_test

import (
	"io"
	"io/ioutil"
	"net/http"
)

type post struct {
	url         string
	bodyType    string
	bodyContent []byte
}

type fakePoster struct {
	posts []post
}

func newFakePoster() *fakePoster {
	return &fakePoster{
		posts: make([]post, 0),
	}
}

func (poster *fakePoster) Post(url string, bodyType string, body io.Reader) (resp *http.Response, err error) {
	bodyContent, _ := ioutil.ReadAll(body)
	poster.posts = append(poster.posts, post{
		url:         url,
		bodyType:    bodyType,
		bodyContent: bodyContent,
	})
	return nil, nil
}
