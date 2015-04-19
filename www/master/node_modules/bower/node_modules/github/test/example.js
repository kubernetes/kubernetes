/** section: github, internal
 *  Example
 * 
 *  Github API usage example.
 * 
 *  Copyright 2012 Cloud9 IDE, Inc.
 *
 *  This product includes software developed by
 *  Cloud9 IDE, Inc (http://c9.io).
 *
 *  Author: Mike de Boer <mike@c9.io>
 **/

"use strict";

var Client = require("./../index");

var github = new Client({
    debug: true,
    version: "3.0.0"
});

github.authenticate({
    type: "basic",
    username: "mikedeboer",
    password: "mysecretpass"
});

github.user.get({}, function(err, res) {
    console.log("GOT ERR?", err);
    console.log("GOT RES?", res);

    github.repos.getAll({}, function(err, res) {
        console.log("GOT ERR?", err);
        console.log("GOT RES?", res);
    });
});
