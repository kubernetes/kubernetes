package main

import (
	"github.com/mjibson/appstats"
)


func stats(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	c := appstats.NewContext(req.Request)
	chain.ProcessFilter(req, resp)
	c.Stats.Status = resp.StatusCode()
	c.Save()
}
