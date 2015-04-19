/**
 *  class Github
 *
 *  A Node.JS module, which provides an object oriented wrapper for the GitHub v3 API.
 *
 *  Copyright 2012 Cloud9 IDE, Inc.
 *
 *  This product includes software developed by
 *  Cloud9 IDE, Inc (http://c9.io).
 *
 *  Author: Mike de Boer <info@mikedeboer.nl>
 **/

"use strict";

var Fs = require("fs");
var Util = require("./../../util");
var error = require("./../../error");

var GithubHandler = module.exports = function(client) {
    this.client = client;
    this.routes = JSON.parse(Fs.readFileSync(__dirname + "/routes.json", "utf8"));
};

var proto = {
    sendError: function(err, block, msg, callback) {
        if (this.client.debug)
            Util.log(err, block, msg.user, "error");
        if (typeof err == "string")
            err = new error.InternalServerError(err);
        if (callback)
            callback(err);
    }
};

["gists", "gitdata", "issues", "authorization", "orgs", "statuses", "pullRequests", "repos", "user", "events", "releases", "search", "markdown", "gitignore", "misc"].forEach(function(api) {
    Util.extend(proto, require("./" + api));
});

GithubHandler.prototype = proto;
